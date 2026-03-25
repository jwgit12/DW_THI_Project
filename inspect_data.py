import argparse
import zarr


def inspect_zarr(zarr_path):
    print(f"Inspecting {zarr_path}\n")
    root = zarr.open_group(zarr_path, mode="r")

    subjects = sorted(k for k in root.keys() if k.startswith("subject_"))
    print(f"Subjects: {len(subjects)}\n")

    for subj_name in subjects:
        subj = root[subj_name]
        bvals = subj["bvals"][:]
        bvecs = subj["bvecs"][:]
        slices = sorted(k for k in subj.keys() if k.startswith("slice_"))

        print(f"── {subj_name} ──")
        print(f"  bvals: shape={bvals.shape}  dtype={bvals.dtype}")
        print(f"  bvecs: shape={bvecs.shape}  dtype={bvecs.dtype}")
        print(f"  slices: {len(slices)}")

        # Show details of first slice as example
        if slices:
            s0 = subj[slices[0]]
            inp = s0["input"]
            tgt = s0["target"]
            msk = s0["mask"]
            print(f"  {slices[0]}/input:  shape={inp.shape}  dtype={inp.dtype}")
            print(f"  {slices[0]}/target: shape={tgt.shape}  dtype={tgt.dtype}")
            print(f"  {slices[0]}/mask:   shape={msk.shape}  dtype={msk.dtype}")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect a DTI ML Zarr dataset")
    parser.add_argument("path", nargs="?", default="dti_ml_dataset_v2.zarr",
                        help="Path to .zarr directory")
    args = parser.parse_args()
    inspect_zarr(args.path)
