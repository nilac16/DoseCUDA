# DoseCUDA

**NOT FOR CLINICAL USE**

**DoseCUDA** is a Python package enabling GPU-based radiation dose calculation for research, development, and education. The package currently supports photon dose calculation using a collapsed cone convolution superposition algorithm and proton dose calculation using a double-Gaussian pencil beam algorithm. The default photon beam model corresponds to the 6 MV energy of a Varian Truebeam linear accelerator and the default proton beam model corresponds to a Hitachi Probeat synchrotron-based PBS delivery system with 98 discrete energies.

# Quickstart Guide

## Prerequisites
Before installing DoseCUDA, ensure you have the following dependencies installed:
- **Python 3.6+**
- **CMake 3.10+**
- **CUDA Toolkit** (Ensure you have a compatible version for your GPU)
- **Git** (for cloning the repository)

Ensure that your GPU and CUDA drivers are properly set up before proceeding.

## Installation on Linux

1. **Install Python** (if not already installed):
   ```
   sudo apt update
   sudo apt install python3 python3-pip
   ```
   
2. **(Optional but recommended)**: Create a virtual environment to isolate DoseCUDA and its dependencies:
   ```
   python3 -m venv dosecuda-env
   source dosecuda-env/bin/activate
   ```

3. **Clone the DoseCUDA repository**:
   ```
   git clone https://github.com/jhu-som-radiation-oncology/DoseCUDA
   cd DoseCUDA
   ```

4. **Install using pip**:
   ```
   python -m pip install .
   ```

5. **Verify the installation**:
   Run the test script to verify that DoseCUDA is installed correctly and working:
   ```
   python tests/test_phantom_impt.py
   ```
   or
   ```
   python tests/test_phantom_imrt.py
   ```

   If everything is installed correctly, you should see the dose calculation output in your terminal and files saved into `./test_phantom_output`.

## Additional Notes:
- **CUDA version**: Ensure that your CUDA version is compatible with your GPU and the CUDA Toolkit installed on your system.
- **Virtual environments**: Virtual environments help avoid conflicts between dependencies required by DoseCUDA and other Python projects you may have on your machine.


6. **Deactivating the virtual environment** (if used):
   After using the package, you can deactivate the virtual environment:
   ```
   deactivate
   ```

## Installation on Windows

1. **Install Python** (if not already installed):
   - Download Python from the [official website](https://www.python.org/downloads/) and install it.
   - Make sure to check the option "Add Python to PATH" during the installation.

2. **Install CMake**:
   - Download the latest version of CMake from the [official website](https://cmake.org/download/).
   - During the installation, select the option to "Add CMake to the system PATH for all users".

3. **Install the CUDA Toolkit**:
   - Download and install the appropriate version of the CUDA Toolkit from the [NVIDIA website](https://developer.nvidia.com/cuda-toolkit).
   - Make sure the drivers for your GPU are compatible with this version.

4. **Install Git**:
   - Download and install Git from the [official website](https://git-scm.com/download/win).
   - During the installation, choose "Git from the command line and also from 3rd-party software".

5. **Create a virtual environment** (optional but recommended):
   Open the command prompt (cmd) and run:
   ```cmd
   python -m venv dosecuda-env
   dosecuda-env\Scripts\activate
   ```

6. **Clone the DoseCUDA repository**:
   ```cmd
   git clone https://github.com/jhu-som-radiation-oncology/DoseCUDA
   cd DoseCUDA
   ```

7. **Install DoseCUDA using pip**:
   ```cmd
   python -m pip install .
   ```

8. **Verify the installation**:
   Run the test script to verify that DoseCUDA is installed correctly and working:
   ```cmd
   python tests/test_phantom_impt.py
   ```

   If everything is installed correctly, the dose calculation output should appear in the terminal, with files saved into `./test_phantom_output`.

9. **Deactivate the virtual environment** (if used):
   After using the package, deactivate the virtual environment by typing:
   ```cmd
   deactivate
   ```

## Additional Notes:
- **PATH configuration**: Ensure Python, CMake, and CUDA are added to your system's PATH during their installations to avoid issues.
- **CUDA compatibility**: Ensure your GPU is compatible with the version of CUDA Toolkit you installed.


# Contact

For any questions or support regarding DoseCUDA, please reach out via email:
* **DoseCUDA@gmail.com**

# Developers

* Tom Hrinivich
* Calin Reamy
* Mahasweta Bhattacharya

# Funding Support
* The Commonwealth Fund
