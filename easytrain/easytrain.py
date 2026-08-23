import os
import gc
import json
import time
import threading
import traceback
import asyncio
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

# Force Hugging Face cache to be local to the execution path
EXECUTION_PATH = Path(os.getcwd())
HF_CACHE_DIR = EXECUTION_PATH / "hf_cache"
ADAPTERS_DIR = EXECUTION_PATH / "adapters"
os.environ["HF_HOME"] = str(HF_CACHE_DIR)

import torch
import reflex as rx
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, Trainer, TrainerCallback,
    TrainingArguments, BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from huggingface_hub import snapshot_download


@dataclass
class TrainConfig:
    model_name: str
    output_name: str
    epochs: float = 1.0
    learning_rate: float = 0.0002
    batch_size: int = 1
    grad_accum: int = 8
    max_seq_len: int = 1024
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    quant: str = '4bit'


class TrainingState:
    """Thread-safe progress state polled by the UI."""
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        with self._lock:
            self.status = 'idle'
            self.step = 0
            self.total_steps = 0
            self.epoch = 0.0
            self.loss_history = deque(maxlen=1000)
            self.logs = deque(maxlen=400)
            self.error = None
            self.start_time = None
            self.stop_requested = False
            self.output_dir = None

    def log(self, msg: str) -> None:
        with self._lock:
            self.logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

    def get_log_string(self) -> str:
        with self._lock:
            return '\n'.join(self.logs)

    def get_status(self) -> str:
        with self._lock:
            return self.status


# Global state manager
STATE = TrainingState()

def tokenize_dataset(file_path, tokenizer, max_seq_len):
    tokenized = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue

            data = json.loads(line)
            prompt     = data.get('input', '')
            completion = data.get('output', '')

            prompt_ids     = tokenizer.encode(prompt,     add_special_tokens=False)
            completion_ids = tokenizer.encode(completion, add_special_tokens=False)  # no BOS on completion

            full_ids = prompt_ids + completion_ids
            labels   = [-100] * len(prompt_ids) + list(completion_ids)  # mask prompt, train on completion

            if len(full_ids) > max_seq_len:
                full_ids = full_ids[:max_seq_len]
                labels   = labels[:max_seq_len]   # truncate labels in sync

            pad = max_seq_len - len(full_ids)
            full_ids = full_ids + [tokenizer.pad_token_id] * pad
            labels   = labels   + [-100] * pad

            tokenized.append({
                'input_ids':      full_ids,
                'attention_mask': [1] * (max_seq_len - pad) + [0] * pad,
                'labels':         labels,
            })

    return tokenized


def _run_training_thread(cfg: TrainConfig, dataset_path: str):
    """Background thread function reproducing the exact training flow."""
    model = trainer = None

    try:
        STATE.log(f'Base model: {cfg.model_name}')

        tokenizer = AutoTokenizer.from_pretrained(
            cfg.model_name,
            cache_dir=HF_CACHE_DIR
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        STATE.log('Tokenizing dataset…')
        tokenized = tokenize_dataset(
            dataset_path,
            tokenizer,
            cfg.max_seq_len
        )

        if len(tokenized) == 0:
            raise ValueError(
                'No usable examples after tokenization — check the dataset format.'
            )

        STATE.log(f'Training on {len(tokenized)} examples.')

        use_cuda = torch.cuda.is_available()
        quant = cfg.quant

        bf16 = use_cuda and torch.cuda.is_bf16_supported()
        compute_dtype = (
            torch.bfloat16
            if bf16
            else torch.float16
            if use_cuda
            else torch.float32
        )

        load_kwargs = {
            'torch_dtype': compute_dtype,
            'cache_dir': HF_CACHE_DIR
        }

        if quant in ['4bit', '8bit']:
            load_kwargs['quantization_config'] = BitsAndBytesConfig(
                load_in_4bit=quant == '4bit',
                load_in_8bit=quant == '8bit',
                bnb_4bit_quant_type='nf4',
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=compute_dtype
            )
            load_kwargs['device_map'] = {'': 0}

        elif use_cuda:
            load_kwargs['device_map'] = {'': 0}

        STATE.log(f'Loading model ({quant} quantization)…')

        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name,
            **load_kwargs
        )

        model.config.use_cache = False

        if quant in ['4bit', '8bit']:
            model = prepare_model_for_kbit_training(model)

        lora = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            bias='none',
            task_type='CAUSAL_LM',
            target_modules='all-linear'
        )

        model = get_peft_model(model, lora)

        trainable, total = model.get_nb_trainable_parameters()

        STATE.log(
            f'LoRA attached: {trainable:,} trainable / '
            f'{total:,} total ({100 * trainable / total:.2f}%).'
        )

        def collate(features):
            return {
                k: torch.tensor([f[k] for f in features], dtype=torch.long)
                for k in ('input_ids', 'attention_mask', 'labels')
            }

        class ProgressCallback(TrainerCallback):
            def on_train_begin(self, args, ts, control, **kw):
                STATE.total_steps = ts.max_steps
                STATE.start_time = time.time()
                STATE.status = 'training'
                STATE.log(
                    f'Training started: {ts.max_steps} steps.'
                )

            def on_log(self, args, ts, control, logs=None, **kw):
                if logs and 'loss' in logs:
                    STATE.step = ts.global_step
                    STATE.epoch = round(ts.epoch or 0, 2)

                    STATE.loss_history.append({
                        'step': ts.global_step,
                        'loss': round(
                            logs.get('loss', 0),
                            4
                        )
                    })

                    STATE.log(
                        f"step {ts.global_step}/{ts.max_steps} "
                        f"loss={logs.get('loss', 0):.4f} "
                        f"lr={logs.get('learning_rate', 0):.2e}"
                    )
                    
                    if len(ts.log_history) > 10:
                        ts.log_history = ts.log_history[-10:]

            def on_step_end(self, args, ts, control, **kw):
                STATE.step = ts.global_step

                if STATE.stop_requested:
                    control.should_training_stop = True

        out_dir = ADAPTERS_DIR / cfg.output_name
        out_dir.mkdir(parents=True, exist_ok=True)

        args = TrainingArguments(
            output_dir=str(out_dir / '_checkpoints'),
            num_train_epochs=cfg.epochs,
            per_device_train_batch_size=cfg.batch_size,
            gradient_accumulation_steps=cfg.grad_accum,
            learning_rate=cfg.learning_rate,
            lr_scheduler_type='cosine',
            warmup_ratio=0.03,
            logging_steps=1,
            save_strategy='no',
            bf16=bf16,
            fp16=use_cuda and (not bf16),
            report_to=[],
            max_grad_norm=1.0,
            remove_unused_columns=False,
            use_cpu=not use_cuda,
            dataloader_pin_memory=use_cuda
        )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=tokenized,
            data_collator=collate,
            callbacks=[ProgressCallback()]
        )

        trainer.train()

        STATE.status = 'saving'
        STATE.log('Saving adapter…')

        model.save_pretrained(out_dir)
        tokenizer.save_pretrained(out_dir)

        # Save simple metadata for merging references later
        meta = asdict(cfg)

        (out_dir / "train_ui_config.json").write_text(
            json.dumps(meta, indent=2)
        )

        STATE.output_dir = str(out_dir)
        STATE.status = (
            'stopped'
            if STATE.stop_requested
            else 'completed'
        )

        STATE.log(
            f"{'Stopped early — adapter saved' if STATE.stop_requested else 'Done!'} "
            f"→ {out_dir}"
        )

    except Exception:
        STATE.error = traceback.format_exc()
        STATE.status = 'failed'

        STATE.log(
            'Training FAILED:\n' + STATE.error
        )

    finally:
        del trainer, model
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Reflex UI interaction layer
# ---------------------------------------------------------------------------

# Reflex upload files are stored separately from the model/dataset mechanics.
# The resulting local path is then passed to the original training function
# exactly as a filesystem path, like the original Gradio file.name.
REFLEX_UPLOAD_DIR = EXECUTION_PATH / ".reflex_uploads"
REFLEX_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def download_action(repo_id):
    if not repo_id.strip():
        return "Please enter a model name."

    try:
        # snapshot_download behavior is unchanged.
        path = snapshot_download(
            repo_id=repo_id,
            cache_dir=HF_CACHE_DIR
        )

        return (
            "✅ Model downloaded successfully to:\n"
            f"{path}"
        )

    except Exception as e:
        return (
            f"❌ Error downloading model: {str(e)}"
        )


def start_training_thread(
    dataset_file: str,
    model_name: str,
    output_name: str,
    epochs: float,
    learning_rate: float,
    batch_size: int,
    grad_accum: int,
    max_seq_len: int,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    quant: str
):
    """
    Exact equivalent of the old start_training setup, except the Gradio
    generator/yield loop is replaced by a Reflex background UI poller.
    """

    if not dataset_file:
        return "❌ Error: Please upload a JSONL dataset."

    cfg = TrainConfig(
        model_name=model_name,
        output_name=output_name,
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        grad_accum=grad_accum,
        max_seq_len=max_seq_len,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        quant=quant
    )

    if not cfg.output_name.strip():
        return "❌ Error: Output name is required."

    STATE.reset()
    STATE.status = 'preparing'

    thread = threading.Thread(
        target=_run_training_thread,
        args=(cfg, dataset_file),
        daemon=True
    )

    thread.start()

    return STATE.get_log_string()


def merge_action(adapter_name):
    """
    The actual merge mechanics are kept the same as the original function.
    This is now a Reflex generator so each yield pushes another UI update.
    """

    adapter_dir = ADAPTERS_DIR / adapter_name

    if not adapter_dir.exists():
        yield (
            f"❌ Error: Adapter directory "
            f"{adapter_dir} not found."
        )
        return

    try:
        config_path = adapter_dir / "train_ui_config.json"

        if not config_path.exists():
            # Fallback PEFT check
            peft_config_path = adapter_dir / "train_ui_config.json"

            if not peft_config_path.exists():
                yield (
                    "❌ Error: Cannot find config to determine base model."
                )
                return

        with open(config_path, 'r') as f:
            cfg = json.load(f)

            base_model = (
                cfg.get("base_model_name_or_path")
                or cfg.get("model_name")
            )

        if not base_model:
            yield (
                "❌ Error: Base model name not found in config."
            )
            return

        yield (
            f"Loading base model {base_model}..."
        )

        use_cuda = torch.cuda.is_available()

        bf16 = (
            use_cuda
            and torch.cuda.is_bf16_supported()
        )

        compute_dtype = (
            torch.bfloat16
            if bf16
            else torch.float16
            if use_cuda
            else torch.float32
        )

        tokenizer = AutoTokenizer.from_pretrained(
            base_model,
            cache_dir=HF_CACHE_DIR
        )

        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=compute_dtype,
            device_map={'': 0} if use_cuda else None,
            cache_dir=HF_CACHE_DIR
        )

        yield (
            f"Loading adapter {adapter_name}..."
        )

        model = PeftModel.from_pretrained(
            model,
            adapter_dir
        )

        yield (
            "Merging adapter into base model "
            "(this may take a while)..."
        )

        model = model.merge_and_unload()

        out_path = adapter_dir / "_merged"
        out_path.mkdir(exist_ok=True)

        yield (
            f"Saving merged model to {out_path}..."
        )

        model.save_pretrained(out_path)
        tokenizer.save_pretrained(out_path)

        yield (
            f"✅ Merge complete! Saved to {out_path}"
        )

    except Exception:
        yield (
            f"❌ Merge Failed: {traceback.format_exc()}"
        )

    finally:
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Reflex State
# ---------------------------------------------------------------------------

class AppState(rx.State):
    # Download
    dl_model_name: str = ""
    dl_output: str = ""

    # Dataset upload
    dataset_file: str = ""
    dataset_filename: str = ""

    # Training inputs
    train_model_name: str = ""
    output_name: str = ""

    epochs: str = "1.0"
    learning_rate: str = "0.0002"
    batch_size: str = "1"
    grad_accum: str = "8"
    max_seq_len: str = "1024"

    lora_r: str = "16"
    lora_alpha: str = "32"
    lora_dropout: str = "0.05"
    quant: str = "4bit"

    # Output
    log_output: str = ""

    # Prevent duplicate UI pollers for the same session.
    polling: bool = False

    @rx.event
    def set_dl_model_name(self, value: str):
        self.dl_model_name = value

    @rx.event
    def set_train_model_name(self, value: str):
        self.train_model_name = value

    @rx.event
    def set_output_name(self, value: str):
        self.output_name = value

    @rx.event
    def set_epochs(self, value: str):
        self.epochs = value

    @rx.event
    def set_learning_rate(self, value: str):
        self.learning_rate = value

    @rx.event
    def set_batch_size(self, value: str):
        self.batch_size = value

    @rx.event
    def set_grad_accum(self, value: str):
        self.grad_accum = value

    @rx.event
    def set_max_seq_len(self, value: str):
        self.max_seq_len = value

    @rx.event
    def set_lora_r(self, value: str):
        self.lora_r = value

    @rx.event
    def set_lora_alpha(self, value: str):
        self.lora_alpha = value

    @rx.event
    def set_lora_dropout(self, value: str):
        self.lora_dropout = value

    @rx.event
    def set_quant(self, value: str):
        self.quant = value

    @rx.event
    def set_dataset_file(self, path: str):
        self.dataset_file = path

    @rx.event
    async def handle_dataset_upload(
        self,
        files: list[rx.UploadFile],
    ):
        """
        Reflex-only upload bridge.

        It saves the selected JSONL to a local path and exposes that path
        to the existing training pipeline. No tokenization/dataset logic
        is changed.
        """
        if not files:
            return

        file = files[0]
        upload_data = await file.read()

        filename = Path(file.name).name
        destination = REFLEX_UPLOAD_DIR / filename

        destination.write_bytes(upload_data)

        self.dataset_file = str(destination)
        self.dataset_filename = filename

        self.log_output = (
            f"Dataset selected: {filename}\n"
            f"Saved to: {destination}"
        )

    @rx.event
    async def download_model(self):
        # Keep the existing synchronous snapshot_download call intact,
        # but execute it off the regular Reflex event path.
        result = await asyncio.to_thread(
            download_action,
            self.dl_model_name,
        )

        self.dl_output = result

    @rx.event
    def start_training(self):
        try:
            result = start_training_thread(
                dataset_file=self.dataset_file,
                model_name=self.train_model_name,
                output_name=self.output_name,
                epochs=float(self.epochs),
                learning_rate=float(self.learning_rate),
                batch_size=int(self.batch_size),
                grad_accum=int(self.grad_accum),
                max_seq_len=int(self.max_seq_len),
                lora_r=int(self.lora_r),
                lora_alpha=int(self.lora_alpha),
                lora_dropout=float(self.lora_dropout),
                quant=self.quant,
            )

            self.log_output = result

            if not self.polling:
                self.polling = True
                return AppState.poll_training

        except Exception:
            self.log_output = (
                "❌ Error starting training:\n"
                f"{traceback.format_exc()}"
            )

    @rx.event(background=True)
    async def poll_training(self):
        """
        Reflex replacement for the Gradio generator loop:

            while thread.is_alive():
                time.sleep(1)
                yield STATE.get_log_string()

        The actual training thread remains exactly the same.
        """
        while True:
            await asyncio.sleep(1)

            log_string = STATE.get_log_string()
            status = STATE.get_status()

            async with self:
                self.log_output = log_string

                if status in [
                    'completed',
                    'failed',
                    'stopped'
                ]:
                    self.polling = False
                    break

        async with self:
            self.log_output = STATE.get_log_string()
            self.polling = False

    @rx.event
    def merge_adapter(self):
        if not self.output_name.strip():
            self.log_output = (
                "❌ Error: Adapter output name is required."
            )
            return

        # Trigger the background event instead of calling it directly.
        return AppState.run_merge(self.output_name)


    @rx.event(background=True)
    async def run_merge(self, adapter_name: str):
        """
        Reflex-only wrapper around the existing merge_action() generator.

        merge_action() itself is unchanged.
        """
        generator = merge_action(adapter_name)

        while True:
            try:
                message = next(generator)
            except StopIteration:
                break

            async with self:
                self.log_output = message

            # Give Reflex a chance to flush the state update.
            await asyncio.sleep(0)

# ---------------------------------------------------------------------------
# Reflex UI
# ---------------------------------------------------------------------------

def text_input(
    label: str,
    value,
    on_change,
    placeholder: str = "",
):
    return rx.vstack(
        rx.text(label, weight="bold"),
        rx.input(
            value=value,
            on_change=on_change,
            placeholder=placeholder,
            width="100%",
        ),
        width="100%",
        align="stretch",
    )


def number_input(
    label: str,
    value,
    on_change,
):
    return rx.vstack(
        rx.text(label, weight="bold"),
        rx.input(
            type="number",
            value=value,
            on_change=on_change,
            width="100%",
        ),
        width="100%",
        align="stretch",
    )


def download_tab():
    return rx.vstack(
        rx.text(
            "Download a base model from Hugging Face "
            "into the local execution path cache."
        ),

        rx.hstack(
            rx.input(
                value=AppState.dl_model_name,
                on_change=AppState.set_dl_model_name,
                placeholder="Qwen/Qwen1.5-1.8B",
                width="100%",
            ),
            rx.button(
                "Download",
                on_click=AppState.download_model,
            ),
            width="100%",
            align="end",
        ),

        rx.text("Status", weight="bold"),
        rx.text_area(
            value=AppState.dl_output,
            read_only=True,
            min_height="120px",
            width="100%",
        ),

        width="100%",
        align="stretch",
    )


def dataset_upload():
    return rx.vstack(
        rx.upload(
            rx.vstack(
                rx.text(
                    "Drop a JSONL dataset here"
                ),
                rx.text(
                    "or click to select a file"
                ),
            ),
            id="dataset_upload",
            accept={
                "application/jsonl": [".jsonl"],
                "application/json": [".jsonl"],
            },
            max_files=1,
            width="100%",
            padding="2em",
            border="1px dashed",
        ),

        rx.button(
            "Upload Dataset",
            on_click=AppState.handle_dataset_upload(
                rx.upload_files(
                    upload_id="dataset_upload"
                )
            ),
        ),

        rx.text(
            rx.cond(
                AppState.dataset_filename != "",
                "Selected: " + AppState.dataset_filename,
                "No dataset selected."
            )
        ),

        width="100%",
        align="stretch",
    )


def training_tab():
    return rx.hstack(
        rx.vstack(
            dataset_upload(),

            text_input(
                "Base Model (Repo ID or local path)",
                AppState.train_model_name,
                AppState.set_train_model_name,
            ),

            text_input(
                "Adapter Output Name",
                AppState.output_name,
                AppState.set_output_name,
            ),

            rx.hstack(
                rx.button(
                    "Start Training",
                    on_click=AppState.start_training,
                ),
                rx.button(
                    "Merge Adapter into Main Model",
                    on_click=AppState.merge_adapter,
                ),
                width="100%",
            ),

            rx.text(
                "Live Training Logs",
                weight="bold",
            ),

            rx.text_area(
                value=AppState.log_output,
                read_only=True,
                min_height="360px",
                width="100%",
            ),

            width="100%",
            align="stretch",
            flex="2",
        ),

        rx.vstack(
            rx.heading(
                "Hyperparameters",
                size="4",
            ),

            number_input(
                "Epochs",
                AppState.epochs,
                AppState.set_epochs,
            ),

            number_input(
                "Learning Rate",
                AppState.learning_rate,
                AppState.set_learning_rate,
            ),

            number_input(
                "Batch Size",
                AppState.batch_size,
                AppState.set_batch_size,
            ),

            number_input(
                "Gradient Accumulation",
                AppState.grad_accum,
                AppState.set_grad_accum,
            ),

            number_input(
                "Max Sequence Length",
                AppState.max_seq_len,
                AppState.set_max_seq_len,
            ),

            rx.heading(
                "LoRA Config",
                size="4",
            ),

            number_input(
                "LoRA Rank (R)",
                AppState.lora_r,
                AppState.set_lora_r,
            ),

            number_input(
                "LoRA Alpha",
                AppState.lora_alpha,
                AppState.set_lora_alpha,
            ),

            number_input(
                "LoRA Dropout",
                AppState.lora_dropout,
                AppState.set_lora_dropout,
            ),

            rx.text(
                "Quantization",
                weight="bold",
            ),

            rx.select(
                ["4bit", "8bit", "none"],
                value=AppState.quant,
                on_change=AppState.set_quant,
                width="100%",
            ),

            width="100%",
            flex="1",
            align="stretch",
        ),

        width="100%",
        align="start",
    )


def index():
    return rx.box(
        rx.vstack(
            rx.heading(
                "Standalone QLoRA Trainer",
                size="6",
            ),

            rx.tabs.root(
                rx.tabs.list(
                    rx.tabs.trigger(
                        "1. Download Model",
                        value="download",
                    ),
                    rx.tabs.trigger(
                        "2. Train Model",
                        value="train",
                    ),
                ),

                rx.tabs.content(
                    download_tab(),
                    value="download",
                    width="100%",
                ),

                rx.tabs.content(
                    training_tab(),
                    value="train",
                    width="100%",
                ),

                default_value="download",
                width="100%",
            ),

            width="100%",
            align="stretch",
        ),

        width="100%",
        max_width="1400px",
        margin="0 auto",
        padding="2rem",
    )


app = rx.App()
app.add_page(
    index,
    route="/",
    title="Standalone QLoRA Trainer",
)


if __name__ == "__main__":
    app.run()