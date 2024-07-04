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
                    element_cnt[site["label"]] += 1

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

total number of entries:  3357619
O: 14.257812%, count: 478723
F: 2.450665%, count: 82284
H: 2.372753%, count: 79668
Al: 2.340260%, count: 78577
S: 2.331265%, count: 78275
P: 2.119002%, count: 71148
Se: 2.116768%, count: 71073
Li: 2.038498%, count: 68445
Si: 1.929760%, count: 64794
Ni: 1.911503%, count: 64181
Cl: 1.812296%, count: 60850
Cu: 1.743587%, count: 58543
Mg: 1.618290%, count: 54336
Co: 1.597680%, count: 53644
Te: 1.577368%, count: 52962
Zn: 1.522478%, count: 51119
In: 1.494601%, count: 50183
Pt: 1.446710%, count: 48575
N: 1.443046%, count: 48452
Rh: 1.402601%, count: 47094
Br: 1.362990%, count: 45764
Ga: 1.350362%, count: 45340
Ca: 1.331152%, count: 44695
As: 1.328412%, count: 44603
Sb: 1.326178%, count: 44528
Ge: 1.319149%, count: 44292
Ag: 1.305419%, count: 43831
Ba: 1.305032%, count: 43818
Fe: 1.300296%, count: 43659
K: 1.283320%, count: 43089
Au: 1.277334%, count: 42888
Sr: 1.232540%, count: 41384
Cd: 1.223843%, count: 41092
La: 1.214879%, count: 40791
Y: 1.178990%, count: 39586
Ir: 1.176399%, count: 39499
B: 1.175238%, count: 39460
Mn: 1.174046%, count: 39420
Zr: 1.132439%, count: 38023
Sc: 1.119484%, count: 37588
Sn: 1.097266%, count: 36842
Pb: 1.083982%, count: 36396
Tl: 1.081332%, count: 36307
Pd: 1.047320%, count: 35165
Na: 1.033113%, count: 34688
Hg: 1.025697%, count: 34439
I: 0.990077%, count: 33243
Ru: 0.986741%, count: 33131
Ta: 0.919640%, count: 30878
Mo: 0.887176%, count: 29788
Ti: 0.841638%, count: 28259
Cr: 0.841400%, count: 28251
C: 0.835116%, count: 28040
Be: 0.776264%, count: 26064
Rb: 0.754404%, count: 25330
Os: 0.744277%, count: 24990
Bi: 0.741508%, count: 24897
W: 0.733704%, count: 24635
Re: 0.667378%, count: 22408
Cs: 0.639918%, count: 21486
Nb: 0.626307%, count: 21029
Tc: 0.623954%, count: 20950
Hf: 0.570017%, count: 19139
Tb: 0.455114%, count: 15281
Dy: 0.442576%, count: 14860
Nd: 0.441503%, count: 14824
Pr: 0.428220%, count: 14378
Er: 0.427059%, count: 14339
Ho: 0.423634%, count: 14224
Tm: 0.406598%, count: 13652
V: 0.403441%, count: 13546
Sm: 0.401683%, count: 13487
Ce: 0.369250%, count: 12398
Lu: 0.310309%, count: 10419
Th: 0.273765%, count: 9192
Eu: 0.216463%, count: 7268
Gd: 0.214050%, count: 7187
Pu: 0.177924%, count: 5974
U: 0.162585%, count: 5459
Np: 0.104657%, count: 3514
Pa: 0.078151%, count: 2624
Pm: 0.039135%, count: 1314
Ac: 0.030885%, count: 1037
Xe: 0.000238%, count: 8
Ne: 0.000030%, count: 1
Kr: 0.000030%, count: 1
Ar: 0.000030%, count: 1
"""