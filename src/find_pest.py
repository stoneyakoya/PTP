from Bio import SeqIO
import requests, re, pandas as pd

def run_epest_local(seq_records):
    with tempfile.TemporaryDirectory() as tmpdir:
        fasta = pathlib.Path(tmpdir, "query.fa")
        with open(fasta, "w") as fh:
            for i, s in enumerate(seq_records):
                fh.write(f">pep{i}\n{textwrap.fill(s, 60)}\n")
        out = pathlib.Path(tmpdir, "out.tsv")
        subprocess.run(["epestfind", "-sequence", fasta, "-window", "10",
                        "-outfile", out], check=True)
        return pd.read_csv(out, sep="\t")


def main():
    df = pd.read_csv("data/dataset/sampling/fold_2/test_fold.csv")
    print(df.columns)
    input()
    for index, row in df.iterrows():
        seq = row["Cleaned_Peptidoform"]
        pest_aa = {"P", "E", "S", "T"}
        if any(aa in seq for aa in pest_aa):
            num, scores = epestifind(seq)
            print(num, scores)
        else:
            print("No PEST found")

if __name__ == "__main__":
    main()