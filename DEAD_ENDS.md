# Dead Ends Log

| Iteration | Milestone | Approach Tried | Why It Failed | Files Touched |
|-----------|-----------|---------------|---------------|---------------|
| Baseline | M1 | Direct static link of kernel.obj into engine.c | engine.c uses dynamic CUDA Driver API (cuModuleLoad) expecting standalone .cubin | build.bat, engine.c |
| Baseline | M1 | Compiling second_brain/legacy_ptq without vcvars64.bat in standard shell | nvcc fatal: cannot find cl.exe in PATH | build.bat |
