# easytrain

Python web UI to train LoRA adapters and merge them into full models. This was the only script that worked for me on Windows 11 CUDA (sm_120).
None of the other methods (Unsloth, llamafactory, etc.) worked in any way without heavy modifications (and then breaking a week later). This should be stable.
It also doesn't require any weird Docker or conda shit.

To run, `run.bat` (it will install packages automatically). To clean cache and rebuild, `clean.bat`.
To delete all your adapters, merged models and huggingface cached models as well, `remove_all_data_and_reinstall.bat` (will also reinstall all packages).

## Notice
I hate Python. Why is there not a C++ library for training LLMs yet??
