# the reason why we have this file is because we want to only train the model on the most common elements in the dataset
# so we maximize the number of trainin gsamples for as few elements (so the model is more accurate for a small number of elements)

from tqdm import tqdm
from collections import Counter
import json
import bz2

IN_DIR = "../../../datasets/alexandria"
only_count_unique_atom_types_per_sample = False

def get_entries(element_cnt, file_path):
    with bz2.open(file_path, 'r') as file_data:
        data = json.load(file_data)
        num_entries = len(data["entries"])
        for i in tqdm(range(num_entries)):
            entry = data["entries"][i]
            if only_count_unique_atom_types_per_sample:
                # count unique atomic numbers since we're only interested in whether an atom type shows up in the sample or not (since it directly determines the size of our training dataset)
                element_cnt.update(entry["data"]["elements"])
            else:
                for site in entry["structure"]["sites"]:
                    element_cnt.update(site["label"])

def main():
    element_cnt = Counter()
    # start at the last file since it's the smallest (faster to parse and iterate)
    for i in range(4, -1, -1):
        file_path = f"{IN_DIR}/alexandria_ps_00{i}.json.bz2"
        get_entries(element_cnt, file_path)

    total_num_entries = sum(element_cnt.values())
    print("total number of entries: ", total_num_entries)

    # print percentages of each element
    sorted_elements = sorted(element_cnt.items(), key=lambda item: item[1], reverse=True)
    for element, count in sorted_elements:
        print(f"{element}: {count / total_num_entries * 100:.6f}%, count: {count}")

if __name__ == "__main__":
    main()

"""
when we set only_count_unique_atom_types_per_sample = True

total number of entries:  1279221
O: 3.448114%, count: 44109
Li: 2.193522%, count: 28060
Al: 2.168742%, count: 27743
Cu: 2.117851%, count: 27092
Ni: 2.021308%, count: 25857
P: 2.011693%, count: 25734
Zn: 1.939696%, count: 24813
Mg: 1.931723%, count: 24711
Si: 1.913274%, count: 24475
Se: 1.901548%, count: 24325
In: 1.883881%, count: 24099
Pt: 1.868168%, count: 23898
Rh: 1.852612%, count: 23699
Co: 1.842371%, count: 23568
Te: 1.768576%, count: 22624
Au: 1.739887%, count: 22257
Sb: 1.732695%, count: 22165
Ag: 1.730897%, count: 22142
Cd: 1.699003%, count: 21734
Ca: 1.695876%, count: 21694
As: 1.690638%, count: 21627
Ba: 1.643735%, count: 21027
Ir: 1.633181%, count: 20892
Cl: 1.628726%, count: 20835
Fe: 1.622784%, count: 20759
Sr: 1.616062%, count: 20673
K: 1.610199%, count: 20598
Mn: 1.583073%, count: 20251
La: 1.578617%, count: 20194
Y: 1.560246%, count: 19959
Tl: 1.559152%, count: 19945
Pb: 1.533590%, count: 19618
Sc: 1.522880%, count: 19481
Hg: 1.498177%, count: 19165
Br: 1.496301%, count: 19141
Zr: 1.469644%, count: 18800
Ru: 1.459091%, count: 18665
S: 1.412188%, count: 18065
N: 1.373648%, count: 17572
Ta: 1.294382%, count: 16558
Ga: 1.260298%, count: 16122
Cr: 1.257719%, count: 16089
Mo: 1.253185%, count: 16031
F: 1.252637%, count: 16024
Ge: 1.243726%, count: 15910
Os: 1.232234%, count: 15763
W: 1.180015%, count: 15095
Sn: 1.178452%, count: 15075
Pd: 1.161097%, count: 14853
Be: 1.108565%, count: 14181
Re: 1.078313%, count: 13794
Tc: 1.072997%, count: 13726
Na: 1.014758%, count: 12981
I: 1.005065%, count: 12857
H: 0.987710%, count: 12635
Rb: 0.907662%, count: 11611
Ti: 0.889526%, count: 11379
C: 0.858257%, count: 10979
B: 0.838323%, count: 10724
Bi: 0.834336%, count: 10673
Cs: 0.757727%, count: 9693
Nb: 0.712543%, count: 9115
Hf: 0.698159%, count: 8931
Nd: 0.432060%, count: 5527
Tb: 0.430731%, count: 5510
Pr: 0.422288%, count: 5402
Dy: 0.421350%, count: 5390
Er: 0.408295%, count: 5223
Ce: 0.407514%, count: 5213
Ho: 0.402823%, count: 5153
Tm: 0.400165%, count: 5119
Sm: 0.397977%, count: 5091
Lu: 0.314723%, count: 4026
Th: 0.304482%, count: 3895
Eu: 0.286190%, count: 3661
V: 0.271884%, count: 3478
Gd: 0.252654%, count: 3232
Pu: 0.242022%, count: 3096
U: 0.198637%, count: 2541
Np: 0.145088%, count: 1856
Pa: 0.104360%, count: 1335
Pm: 0.066134%, count: 846
Ac: 0.059098%, count: 756
Xe: 0.000235%, count: 3
Ne: 0.000078%, count: 1
Kr: 0.000078%, count: 1
Ar: 0.000078%, count: 1
"""


"""
when we set only_count_unique_atom_types_per_sample = False

total number of entries:  5649630
O: 8.915858%, count: 503713
S: 6.867193%, count: 387971
C: 6.177378%, count: 348999
e: 4.829785%, count: 272865
i: 4.435264%, count: 250576
a: 4.298228%, count: 242834
r: 3.923089%, count: 221640
P: 3.815719%, count: 215574
A: 3.733643%, count: 210937
T: 3.672471%, count: 207481
N: 3.304446%, count: 186689
B: 3.186102%, count: 180003
n: 3.142931%, count: 177564
l: 3.110540%, count: 175734
u: 2.800590%, count: 158223
H: 2.610259%, count: 147470
b: 2.523422%, count: 142564
g: 2.347163%, count: 132606
R: 2.264980%, count: 127963
F: 2.229226%, count: 125943
M: 2.186763%, count: 123544
I: 2.175806%, count: 122925
L: 2.117926%, count: 119655
d: 1.739371%, count: 98268
o: 1.728538%, count: 97656
G: 1.713723%, count: 96819
s: 1.612123%, count: 91079
Z: 1.577838%, count: 89142
c: 1.054494%, count: 59575
h: 0.996278%, count: 56286
t: 0.859791%, count: 48575
K: 0.762705%, count: 43090
Y: 0.700683%, count: 39586
m: 0.503626%, count: 28453
W: 0.436046%, count: 24635
E: 0.382450%, count: 21607
f: 0.338766%, count: 19139
D: 0.263026%, count: 14860
y: 0.263026%, count: 14860
V: 0.239768%, count: 13546
U: 0.096626%, count: 5459
p: 0.062199%, count: 3514
X: 0.000142%, count: 8
"""