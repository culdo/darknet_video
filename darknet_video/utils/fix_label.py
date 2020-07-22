import glob
import os


def fix_label(dir):
    for fname in glob.glob(dir):
        label = int(os.path.basename(fname).split("_")[0])
        with open(fname, "r") as f:
            line = f.readline()
        line = line.replace("%d " % label, "%d " % (label-1))
        print(line)
        with open(fname, "w") as f:
            f.write(line)


if __name__ == '__main__':
    fix_label("/home/lab-pc1/nptu/lab/computer_vision/darknet/data/hand_test/val/*.txt")
