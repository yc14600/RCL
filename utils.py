import os
import csv

def config_path(rpath):
    try:
        os.makedirs(rpath)
    except FileExistsError: 
        return rpath
    
    return rpath

def config_result_path(args):
    rpath = config_path(args.result_path)
    rpath = config_path(os.path.join(rpath,args.benchmark))
    rpath = config_path(os.path.join(rpath,args.strategy_type+'_'+args.replay_strategy))

    folder_name = args.benchmark+'_'+args.strategy_type\
                    +'_'+args.model_type+'_T'+str(args.T)\
                    +'_ep'+str(args.epoch)+'_sd'+str(args.seed)
                    
    rpath = os.path.join(rpath,folder_name)
                    
    
    dpath = config_path(os.path.join(rpath,'decoder'))
    config_path(os.path.join(dpath,'images_previous'))
    config_path(os.path.join(dpath,'images_after'))

    with open(os.path.join(rpath,'configures.txt'),'w') as f:
        f.write(str(args))

    return rpath,dpath


def str2bool(x):
    if x.lower() == 'false':
        return False
    else:
        return True
    

def representation_log(t,reps,lbls,path):
    
    with open(os.path.join(path,'rep_log'+'_t'+str(t)+'.txt'),'a') as f:
        writer = csv.writer(f,delimiter=',')
        for l in set(lbls.numpy()):
            writer.writerow(['label',l])
            lrp = reps[lbls==l]
            writer.writerow(['rep mean']+[lrp.mean(axis=0)])
            writer.writerow(['rep std']+[lrp.std(axis=0)])