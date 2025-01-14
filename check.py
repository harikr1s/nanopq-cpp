import filecmp

f1 = "out/py_encoded.txt"
f2 = "out/cpp_encoded.txt"

if filecmp.cmp(f1, f2, shallow=False):
	print("Encoding verified!")
else:
	print("Encoding error!")
