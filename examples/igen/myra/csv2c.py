import vu_common as CM

def read_csv(csv_file):
    import csv    
    with open(csv_file, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for i,row in enumerate(reader):
            if i == 0:
                myvars = row[:-1]
                stmts = ["int {} = atoi(argv[{}]);".format(v,j+1)
                         for j,v in enumerate(myvars)]
                continue
            
            configs = zip(myvars,row[:-1])
            outp = row[-1]
            stmt = ' && '.join("{}=={}".format(a,b) for a,b in configs)
            stmt = "if ({}) printf(\"{}\\n\");".format(stmt,outp)
            stmts.append(stmt)
                
        return stmts


if __name__ == "__main__":
    csv_file = "invariants.csv"    
    stmts = read_csv(csv_file)
    print('\n'.join(stmts))
    
