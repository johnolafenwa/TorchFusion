from torchtext.data import TabularDataset,BucketIterator
import json
import os



def load_tabular_set(file_path,format,fields,split_ratio=None,split_seed=None,skip_header=False,save_vocab_path=os.getcwd(),**args):

    """

    :param file_path:
    :param format:
    :param fields:
    :param split_ratio:
    :param split_seed:
    :param skip_header:
    :param save_vocab_path:
    :param args:
    :return:
    """
    if os.path.exists(save_vocab_path) == False:
        os.mkdir(save_vocab_path)

    dataset_fields = []

    for field in fields:
        dataset_fields.append((field.name,field.field))

    dataset = TabularDataset(file_path,format,dataset_fields,skip_header=skip_header,**args)

    for f_input in fields:
        name = f_input.name
        field = f_input.field
        vocab = f_input.vocab

        if vocab is None:

            field.build_vocab(dataset,max_size=f_input.max_size, min_freq=f_input.min_freq,
                 vectors=f_input.vectors, unk_init=f_input.unk_init, vectors_cache=f_input.vectors_cache)

            with open(os.path.join(save_vocab_path,"{}.json".format(name)), "w") as jfile:
                json.dump(field.vocab.stoi,jfile,sort_keys=True)

        else:
            with open(vocab, "r") as jfile:
                dict_ = json.load(jfile)

                field.build_vocab()
                field.vocab.stoi = dict_



    if split_ratio is not None:

        dataset = dataset.split(split_ratio,random_state=split_seed)

    return dataset




def load_tabular_set_split(root_path,format,fields,train=None,val=None,test=None,skip_header=False,save_vocab_path=os.getcwd(),**args):
    """

    :param root_path:
    :param format:
    :param fields:
    :param train:
    :param val:
    :param test:
    :param skip_header:
    :param save_vocab_path:
    :param args:
    :return:
    """
    if os.path.exists(save_vocab_path) == False:
        os.mkdir(save_vocab_path)

    dataset_fields = []

    for field in fields:
        dataset_fields.append((field.name,field.field))
    print(dataset_fields)
    dataset = TabularDataset.splits(root_path,".data",train,val,test,fields=dataset_fields,skip_header=skip_header,format=format,**args)

    for f_input in fields:
        name = f_input.name
        field = f_input.field
        vocab = f_input.vocab

        if vocab is None:
            #verify if working properly
            field.build_vocab(*dataset,max_size=f_input.max_size, min_freq=f_input.min_freq,
                 vectors=f_input.vectors, unk_init=f_input.unk_init, vectors_cache=f_input.vectors_cache)

            with open(os.path.join(save_vocab_path,"{}.json".format(name)), "w") as jfile:
                json.dump(field.vocab.stoi,jfile,sort_keys=True)

        else:
            with open(vocab, "r") as jfile:
                dict_ = json.load(jfile)
                field.build_vocab()
                field.vocab.stoi = dict_


    return dataset


def csv_data(file_path,fields,split_ratio=None,split_seed=None,skip_header=False,save_vocab_path=os.getcwd(),**args):
    """

    :param file_path:
    :param fields:
    :param split_ratio:
    :param split_seed:
    :param skip_header:
    :param save_vocab_path:
    :param args:
    :return:
    """

    return load_tabular_set(file_path,"csv",fields=fields,split_ratio=split_ratio,split_seed=split_seed,skip_header=skip_header,save_vocab_path=save_vocab_path,**args)

def csv_data_split(root_path,fields,train,val=None,test=None,skip_header=False,save_vocab_path=os.getcwd(),**args):
    """

    :param root_path:
    :param fields:
    :param train:
    :param val:
    :param test:
    :param skip_header:
    :param save_vocab_path:
    :param args:
    :return:
    """
    return load_tabular_set_split(root_path,"csv",fields=fields,train=train,val=val,test=test,skip_header=skip_header,save_vocab_path=save_vocab_path,**args)

def tsv_data(file_path,fields,split_ratio=None,split_seed=None,skip_header=False,save_vocab_path=os.getcwd(),**args):
    """

    :param file_path:
    :param fields:
    :param split_ratio:
    :param split_seed:
    :param skip_header:
    :param save_vocab_path:
    :param args:
    :return:
    """

    return load_tabular_set(file_path,"tsv",fields=fields,split_ratio=split_ratio,split_seed=split_seed,skip_header=skip_header,save_vocab_path=save_vocab_path,**args)

def tsv_data_split(root_path,fields,train,val=None,test=None,skip_header=False,save_vocab_path=os.getcwd(),**args):
    """

    :param root_path:
    :param fields:
    :param train:
    :param val:
    :param test:
    :param skip_header:
    :param save_vocab_path:
    :param args:
    :return:
    """
    return load_tabular_set_split(root_path,"tsv",fields=fields,train=train,val=val,test=test,skip_header=skip_header,save_vocab_path=save_vocab_path,**args)


def json_data(file_path,fields,split_ratio=None,split_seed=None,skip_header=False,save_vocab_path=os.getcwd(),**args):
    """

    :param file_path:
    :param fields:
    :param split_ratio:
    :param split_seed:
    :param skip_header:
    :param save_vocab_path:
    :param args:
    :return:
    """

    return load_tabular_set(file_path,"json",fields=fields,split_ratio=split_ratio,split_seed=split_seed,skip_header=skip_header,save_vocab_path=save_vocab_path,**args)

def json_data_split(root_path,fields,train,val=None,test=None,skip_header=False,save_vocab_path=os.getcwd(),**args):
    """

    :param root_path:
    :param fields:
    :param train:
    :param val:
    :param test:
    :param skip_header:
    :param save_vocab_path:
    :param args:
    :return:
    """
    return load_tabular_set_split(root_path,"json",fields=fields,train=train,val=val,test=test,skip_header=skip_header,save_vocab_path=save_vocab_path,**args)



def csv_data_loader(file_path,fields,split_ratio=None,split_seed=None,skip_header=False,save_vocab_path=os.getcwd(),batch_size=32,device=None,train=True,**args):
    """

    :param file_path:
    :param fields:
    :param split_ratio:
    :param split_seed:
    :param skip_header:
    :param save_vocab_path:
    :param batch_size:
    :param device:
    :param train:
    :param args:
    :return:
    """
    dataset = load_tabular_set(file_path,"csv",fields=fields,split_ratio=split_ratio,split_seed=split_seed,skip_header=skip_header,save_vocab_path=save_vocab_path,**args)
    return BucketIterator(dataset,batch_size=batch_size,device=device,train=True,shuffle=train,repeat=False)

def csv_data_split_loader(root_path,fields,train=None,val=None,test=None,skip_header=False,save_vocab_path=os.getcwd(),batch_size=32,device=None,**args):
    """

    :param root_path:
    :param fields:
    :param train:
    :param val:
    :param test:
    :param skip_header:
    :param save_vocab_path:
    :param batch_size:
    :param device:
    :param args:
    :return:
    """
    dataset = load_tabular_set_split(root_path,"csv",fields=fields,train=train,val=val, test=test,skip_header=skip_header,save_vocab_path=save_vocab_path,**args)
    return BucketIterator(dataset, batch_size=batch_size, device=device, train=True, shuffle=train,repeat=False)

def tsv_data_loader(file_path,fields,split_ratio=None,split_seed=None,skip_header=False,save_vocab_path=os.getcwd(),batch_size=32,device=None,train=True,**args):
    """

    :param file_path:
    :param fields:
    :param split_ratio:
    :param split_seed:
    :param skip_header:
    :param save_vocab_path:
    :param batch_size:
    :param device:
    :param train:
    :param args:
    :return:
    """
    dataset = load_tabular_set(file_path,"tsv",fields=fields,split_ratio=split_ratio,split_seed=split_seed,skip_header=skip_header,save_vocab_path=save_vocab_path,**args)
    return BucketIterator(dataset, batch_size=batch_size, device=device, train=True, shuffle=train,repeat=False)

def tsv_data_split_loader(root_path,fields,train=None,val=None,test=None,skip_header=False,save_vocab_path=os.getcwd(),batch_size=32,device=None,**args):
    """

    :param root_path:
    :param fields:
    :param train:
    :param val:
    :param test:
    :param skip_header:
    :param save_vocab_path:
    :param batch_size:
    :param device:
    :param args:
    :return:
    """
    dataset = load_tabular_set_split(root_path,"tsv",fields=fields,train=train,val=val,test=test,skip_header=skip_header,save_vocab_path=save_vocab_path,**args)
    return BucketIterator(dataset, batch_size=batch_size, device=device, train=True, shuffle=train,repeat=False)

def json_data_loader(file_path,fields,split_ratio=None,split_seed=None,skip_header=False,save_vocab_path=os.getcwd(),batch_size=32,device=None,train=True,**args):
    """

    :param file_path:
    :param fields:
    :param split_ratio:
    :param split_seed:
    :param skip_header:
    :param save_vocab_path:
    :param batch_size:
    :param device:
    :param train:
    :param args:
    :return:
    """
    dataset = load_tabular_set(file_path,"json",fields=fields,split_ratio=split_ratio,split_seed=split_seed,skip_header=skip_header,save_vocab_path=save_vocab_path,**args)
    return BucketIterator(dataset, batch_size=batch_size, device=device, train=True, shuffle=train,repeat=False)

def json_data_split_loader(root_path,fields,train=None,val=None,test=None,skip_header=False,save_vocab_path=os.getcwd(),batch_size=32,device=None,**args):
    """

    :param root_path:
    :param fields:
    :param train:
    :param val:
    :param test:
    :param skip_header:
    :param save_vocab_path:
    :param batch_size:
    :param device:
    :param args:
    :return:
    """
    dataset = load_tabular_set_split(root_path,"json",fields=fields,train=train,val=val,test=test,skip_header=skip_header,save_vocab_path=save_vocab_path,**args)
    return BucketIterator(dataset, batch_size=batch_size, device=device, train=True, shuffle=train,repeat=False)
