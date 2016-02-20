def main(s, t, u, v, x ,y , z):
    #options: s,t,u,v, x,y: bool
    #z: {0,..,4}

    max_z = 3

    if (x and y):
        print "L0" #x & y
        if not (0 < z  < max_z):
            print "L1" #x & y & (z=0|3|4)

    else:
        print "L2" # !x|!y
        print "L2c" # !x|!y        

    print "L3" #true
    if u and v:
        print "L4" #u && v
        if s or t:
            print "L5" # (s|t) & (u&v)
            return

    print "L6" # (s=0 && t=0) | (u=0 | v=0)


if __name__ == "__main__":
    import argparse
    aparser = argparse.ArgumentParser()
    aparser.add_argument("-s", type=int, choices=range(2))
    aparser.add_argument("-t", type=int, choices=range(2))
    aparser.add_argument("-u", type=int, choices=range(2))
    aparser.add_argument("-v", type=int, choices=range(2))
    aparser.add_argument("-x", type=int, choices=range(2))
    aparser.add_argument("-y", type=int, choices=range(2))
    aparser.add_argument("-z", type=int, choices=range(5))
                         
    args = aparser.parse_args()
    main(args.s, args.t, args.u, args.v, args.x, args.y, args.z)

