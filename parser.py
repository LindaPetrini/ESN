
# HOWTO:
# PUT data to be parsed in ./data/
# CREATE a directory ./output/

from os import listdir
from os.path import isfile, join

def parse(inp, output):
    lines = []
    n_lines = 0
    with open('./' + inp, 'r') as f:
        for line in f:
            n_lines += 1
    
    print("File len: ", n_lines)
    with open('./' + inp, 'r') as f:
        for line in f:
            headers = []
            parsed_line = []
            line = line.strip()[1:-1]
            data = line.split(")(")
            for d in data:
                head, values = d.split(' ', 1)
                values = values.split()
                for idx, v in enumerate(values):
                    headers.append(head+"_"+str(idx))
                    parsed_line.append(str(v))
            lines.append(",".join(parsed_line)+"\n")
        headers = ",".join(headers)+"\n"
    
    ("Parsing Complete...")
    with open('./' + output, 'w') as f:
        f.write(headers)
        for line in lines:
            f.write(line)

myfiles = [f for f in listdir("./data/") if isfile(join("./data/", f))]
inpath = "./data/"
outpath = "./output/"
#print(onlyfiles)

for inp in myfiles:
    parse(inpath+inp, outpath+inp)
    print("Parsing of ", inp, "completed!")
