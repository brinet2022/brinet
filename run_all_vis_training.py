from vis_training import main as vis

class argsP():
    def __init__(self, exp_num, max_fold):
        self.exp_num=exp_num
        self.max_fold=max_fold
        self.stats_only=True
        self.root_path = "./visualization/training"
def main():
    #exp = list(range(15003, 15006)) + list(range(15009, 15012))
    #exp = list(range(12100, 12106)) + list(range(12600, 12606))
    exp = list(range(15000, 15005)) + list(range(15006, 15012))
    for i in exp:
        print ("exp: {}".format(i))
        args  = argsP(i, 1)
        vis(args)
    #for i in range(600, 840):
    #    print ("exp: {}".format(i))
    #    args  = argsP(i, 5)
    #    vis(args)

main()
