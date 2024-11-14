from DoseCUDA import IMPTDoseGrid, IMPTPlan
import subprocess
import os
import time
import sys
import json


# Figure out what to do
# Returns a tuple of (test-dir, output-path)
def parse_args(argv):
    def print_usage():
        print("Supply a path to a test directory, and optionally a name for the"
              " output file.\nThe default output filename is \"output.nrrd\"")

    if len(argv) < 2:
        print_usage()
        return (None, None)

    if len(argv) > 2:
        output = argv[2]
    else:
        output = "output.nrrd"

    return argv[1], os.path.join(argv[1], output)


# Open a directory containing a JSON file with paths to the test input files
# Return a tuple of (RD-path, RP-path, CT-path)
def load_test_directory(dir: str):
    with open(os.path.join(dir, "test.json")) as fp:
        js = json.load(fp)
    tup = js["rtdose"], js["rtplan"], js["ct"]
    return (os.path.join(dir, file) for file in tup)


# Inspect the filename, and dispatch to the correct loader based on extension
# (yikes)
def load_CT(dose, ct):
    base, ext = os.path.splitext(ct)
    if ext.lower() == ".nrrd":
        dose.loadCTNRRD(ct)
    else:
        # Should test to see if it's a directory
        dose.loadCTDCM(ct)


# Determine the output type and dispatch to the appropriate method
def write_dose(dose, rd, path):
    base, ext = os.path.splitext(path)
    if ext.lower() == ".nrrd":
        dose.writeDoseNRRD(path, individual_beams=False, dose_type="PHYSICAL")
    else:
        dose.writeDoseDCM(path, rd, dose_type="PHYSICAL")


# Use Plastimatch to check the gamma
def test_plastimatch(ref, test):
    subprocess.run(["plastimatch",
                    "gamma",
                    "--inherent-resample",
                    "2.0",
                    ref,
                    test])


# Use my own pattern search to check the gamma
def test_mu(ref, test):
    build = os.path.expanduser("~/Source/repos/C++/MU/build")
    exe = os.path.join(build, "test/gamma")
    mods = os.path.join(build, "modules")
    subprocess.run([exe, "-m", mods, "--", ref, test])


if __name__ == "__main__":
    dir, output = parse_args(sys.argv)
    if dir is None:
        exit(1)

    plan = IMPTPlan()
    dose = IMPTDoseGrid()
    rd, rp, ct = load_test_directory(dir)

    print("Reading plan...")
    plan.readPlanDicom(rp)

    print("Reading CT...")
    load_CT(dose, ct)

    print("Resampling CT...")
    dose.resampleCTfromReferenceDose(rd)

    print("Computing dose...")
    tic = time.time()
    dose.computeIMPTPlan(plan, gpu_id=0)    # Why GPU 1?
    toc = time.time()
    print("Time elapsed: %.2f s" % (toc - tic))

    print("Writing dose...")
    write_dose(dose, rd, output)

    print("Quick gamma...")
    test_mu(rd, output)

    print("Done.")
