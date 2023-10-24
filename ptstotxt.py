import os
import struct

def pts2txt(din, dout, src):
    src_p = os.path.join(din, src)
    data = open(src_p, 'rb').read()
    if len(data) < 692:
        print(str(src) + ' is broken - ', len(data))
        return 0
    points = struct.unpack('i172f', data)

    dst = src.replace('pts', 'txt')
    dst_p = os.path.join(dout, dst)

    with open(dst_p, 'w') as fout:
        pnum = len(points[1:])
        for i in range(1, pnum, 2):
            fout.write('%f ' % points[i])
            fout.write('%f\n' % points[i+1])

    return 1

def main():
    input_dir = 'data/landmarks'
    output_dir = 'data/landmarks'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith('.pts'):
            pts2txt(input_dir, output_dir, filename)

if __name__ == "__main__":
    main()
