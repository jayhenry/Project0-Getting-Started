# CIS 5650 CUDA Introduction

The CUDA Introduction exercises are design to get you accustomed to CUDA Programming using simple operations:

1. SAXPY (`z[] = a * x[] + y[]`)
2. Matrix Transpose
3. Matrix Multiplication

## Clone, Build and Run

Clone this repository from Github and then run the instructions below.

1. From the `cuda-introduction` directory, run:
    * Windows: `cmake -B build -S . -G "Visual Studio 17 2022"` to generate the Visual Studio project. You may choose a different Visual Studio version.
    * Linux: `cmake -B build -S . -G "Unix Makefiles"` to generate the makefiles.
2. Open the generated `CUDAIntroduction` project from the `build` directory.
3. Build the project in Visual Studio. (Note that there are Debug and Release configuration options.)
4. Run. Make sure you run the actual project as target (not `ALL_BUILD`) by right-clicking it and selecting "Set as StartUp Project".

Please ask me or the TAs ahead of time if you have trouble compiling the code. We want to be ready to go at the start of the lab.

## Complete Intoduction Exercises

Start with SAXPY, then Transpose, then Matrix Multiplication. In each file, follow the `TODO`s, which are numbered in order.

The `LOOK` comments are designed to show you best practices for CUDA Programming. You can copy these snippets into future projects if needed.

Once you have completed each the the exercises, follow the `TODO Optional`s for testing different configurations of sizes.

### Use Nsight Debuggers

With each implementation, run the NSight Debugger. Walk through the steps, go to different threads, warps, blocks, inspect the variables. Try to thoroughly understand your code as well as the debugging tools available to you.

### Unblocking

If you get stuck at any point, follow this order:

1. Use the Nsight debugger to understand the problem. Use paper and pencil to write down the equations especially with regards to indexing.
2. Search on the internet. Try to understand the code others have written and use that to solve your problem.
3. Use the [cuda-introduction-solutions branch](https://github.com/CIS5650-Fall-2025/Project0-Getting-Started/tree/cuda-introduction-solutions/cuda-introduction). The solutions for all the exercises are provided. For best learning, try to solve the problems on your own and only use this as reference to compare your implementation.

### Visual Studio Hint:

* If you start running using `F5`, the command prompt will open and close.
    * The `F5` shortcut is *Start Debugging*, which means Visual Studio will monitoring your application and it will not run at full performance.
    * Use `F5` only when you are debugging.
* Instead, use `Ctrl+F5` when you want to run without debugging. This will run the application at full performance as well as keep the command prompt open after the application ends.

## Third Party Code

This repository includes code from [termcolor](https://github.com/ikalnytskyi/termcolor) licensed under the BSD 3 Clause.
