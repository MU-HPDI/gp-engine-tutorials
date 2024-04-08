import torch
import sys


def matmul(size=50):

    mat1 = torch.randn(size, size).cuda()
    mat2 = torch.randn(size, size).cuda()

    torch.matmul(mat1, mat2)

    print("Done!")


if __name__ == "__main__":
    size = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    matmul(size=size)
