import os
import sys
import glob
import torch
import traceback

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from xgutils import sysutil, nputil, ptutil, visutil

# Pytorch Lightning
from pytorch_lightning import Callback, LightningModule, Trainer
def dataset_generator(pl_module, dset, data_indices=[0,1,2], **get_kwargs):
    is_training = pl_module.training
    with torch.no_grad():
        pl_module.eval()
        for ind in data_indices:
            dataitem = dset.__getitem__(ind,**get_kwargs)
            batch = {}
            for key in dataitem:
                datakey = dataitem[key]
                if type(datakey) is not np.ndarray and type(datakey) is not torch.Tensor:
                    continue
                datakey = dataitem[key][None,...]
                if type(datakey) is np.ndarray:
                    datakey = torch.from_numpy(datakey)
                batch[key] = datakey.to(pl_module.device)
            yield batch
        if is_training:
            pl_module.train()
        else:
            pl_module.eval()

from abc import ABC, abstractmethod
from pytorch_lightning import Callback, LightningModule, Trainer

class FlyObj():
    def __init__(self, save_dir=None, load_dir=None, on_the_fly=True, data_processor=None):
        if data_processor is None:
            data_processor = self.dflt_data_processor
        self.__dict__.update(locals())
    def process_iter(self, input_iter):
        for name, input_data in input_iter:
            processed = self.load(name)
            if processed is None:
                processed = self.data_processor(input_data, input_name=name)
            yield name, processed

    def __call__(self, input_iter):
        process_iter = self.process_iter(input_iter)
        
        if self.on_the_fly==False:
            all_processed = list(process_iter)
            list(starmap(self.save, all_processed))
            for name, processed in all_processed:
                yield name, processed
        else:
            for name, processed in process_iter:
                self.save(name, processed)
                yield name, processed
    @staticmethod
    def dflt_data_processor(input_data):
        return input_data
    def save(self, name, data):
        if self.save_dir is not None:
            sysutil.mkdirs(self.save_dir)
            save_path = os.path.join(self.save_dir, f"{name}.npy")
            np.save(save_path, ptutil.ths2nps(data))
    def load(self, name):
        if self.load_dir is None:
            return None
        load_path = os.path.join(self.load_dir, f"{name}.npy")
        if os.path.exists(load_path) == False:
            return None
        loaded    = np.load(load_path,allow_pickle=True).item()
        return loaded
class ImageFlyObj(FlyObj):
    def save(self, name, imgs):
        if self.save_dir is not None:
            sysutil.mkdirs(self.save_dir)
            for key in imgs:
                save_path = os.path.join(self.save_dir, f"{name}_{key}.png")
                visutil.saveImg(save_path, imgs[key])
    def load(self, name):
        if self.load_dir is None:
            return None
        load_paths = os.path.join(self.load_dir, f"{name}_*.png")
        files = glob.glob(load_paths)
        files.sort(key=os.path.getmtime)
        if len(files) == 0:
            return None
        loaded = {}
        for imgf in files:
            key = "_".join(imgf[:-4].split("_")[1:])
            loaded[key] = visutil.readImg(imgf)
        return loaded
def dataset_generator(pl_module, dset, data_indices=[0,1,2], yield_ind=True, **get_kwargs):
    is_training = pl_module.training
    with torch.no_grad():
        pl_module.eval()
        for ind in data_indices:
            dataitem = dset.__getitem__(ind,**get_kwargs)
            batch = {}
            for key in dataitem:
                datakey = dataitem[key]
                if type(datakey) is not np.ndarray and type(datakey) is not torch.Tensor:
                    continue
                datakey = dataitem[key][None,...]
                if type(datakey) is np.ndarray:
                    datakey = torch.from_numpy(datakey)
                batch[key] = datakey.to(pl_module.device)
            if yield_ind==True:
                yield str(ind), batch
            else:
                yield batch
        if is_training:
            pl_module.train()
        else:
            pl_module.eval()

def get_effective_visual_indices(indices, global_rank, gpu_nums):
    """ Assign different indices for different gpus
        global_rank: which gpu is the current device
        gpu_nums: total number of gpus
        [[0 5]
         [1 6]
         [2 7]
         [3]
         [4]
        ]
    """
    indices = np.array(indices)
    total_num = len(indices)
    batch_size = nputil.ceiling_division(total_num-global_rank, gpu_nums)
    effective_ind =  global_rank + gpu_nums * np.arange(batch_size)
    effective = indices[effective_ind]
    return effective
def get_effective_visual_indices_unittest():
    gpus_nums=5
    for rg in range(20):
        for i in range(gpus_nums):
            x=get_effective_visual_indices( np.arange(rg)*2, i, gpus_nums)
            print(x)
        print("!")
class VisCallback(Callback):
    def __init__(self,  visual_indices=[0,1,2,3,4,5], all_indices=False, force_visual_indices=False, \
                        every_n_epoch=3, no_sanity_check=False, \
                        load_compute=False, load_visual=False, \
                        data_dir = None, output_name=None, use_dloader=False, num_gpus=1, 
                        parallel_vis=False, single_vis=True, visall_after_training_end=True):
        super().__init__()
        self.__dict__.update(locals())
        self.classname = self.__class__.__name__
        if self.output_name == None:
            self.output_name = self.classname
        if self.data_dir is None:
            self.data_dir = f"/studio/nnrecon/temp/{self.output_name}/"
        if all_indices is True and force_visual_indices==False:
            self.visual_indices = "all"
        
    def process(self, pl_module, dloader, visual_indices=None, data_dir=None, visual_summary=False, \
                parallel_vis=None, load_compute=None, load_visual=None, fly_compute=True):
        self.pl_module = pl_module
        dset = dloader.dataset
        if load_compute is None:
            load_compute = self.load_compute
        if load_visual is None:
            load_visual = self.load_visual
        if data_dir is None:
            data_dir = self.data_dir
        if visual_indices is None:
            visual_indices = self.visual_indices
        if parallel_vis is None:
            parallel_vis = self.parallel_vis
        if visual_indices is "all":
            visual_indices = list(range(len(dset)))
        print("is force?", self.force_visual_indices)
        if self.force_visual_indices==True:
            visual_indices = self.visual_indices
        print("self.visual_indices", self.visual_indices)
        compute_dir = os.path.join(data_dir, "computed")
        cload_dir   = compute_dir if load_compute==True else None
        visual_dir  = os.path.join(data_dir, "visual")
        vload_dir   = visual_dir  if load_visual==True  else None
        if parallel_vis==True:
            visual_indices = get_effective_visual_indices(visual_indices, pl_module.global_rank, self.num_gpus)
            print("global_rank", self.pl_module.global_rank, "/", self.num_gpus, "effecitve indices", visual_indices)

        if self.use_dloader==False:
            datagen      = dataset_generator(pl_module, dset, visual_indices)
        else:
            datagen      = dloader
        computegen   = FlyObj(data_processor=self.compute_batch, save_dir=compute_dir, load_dir=cload_dir, on_the_fly=fly_compute)
        visgen       = ImageFlyObj(data_processor=self.visualize_batch, save_dir=visual_dir, load_dir=vload_dir)
        imgsgen,imgs = visgen(computegen(datagen)), []
        failed_ind = []
        for ind in sysutil.progbar(visual_indices):
            try:
                imgs.append(next(imgsgen))
            except Exception as e:
                traceback.print_exc()
                print(e)
                failed_ind.append(ind)
        failed_log = os.path.join(self.data_dir, f"logs/failed_ind/")
        sysutil.mkdirs(failed_log)
        np.savetxt( failed_log+f"/rank_{self.pl_module.global_rank}.txt", 
                    np.array(failed_ind))

        if visual_summary==True:
            summary_imgs = self.get_summary_imgs(imgs, zoomfac=.5)
        else:
            summary_imgs = None
        self.imgs, self.summary_imgs = imgs, summary_imgs
        
        #for l,img in visgen(computegen(datagen)):
        #    visutil.showImg(img["recon"])
        #visutil.showImg(self.summary_imgs[self.summary_imgs.keys()[0]]["image"])
        #return self.summary_imgs
    def process_all(self, pl_module, dloader, parallel_vis=True, **kwargs):
        return self.process(pl_module, dloader, parallel_vis=parallel_vis, visual_indices="all", **kwargs)
    def compute_batch(batch):
        logits = batch["Ytg"].clone()
        logits[:]= torch.rand(logits.shape[0])
        return {"logits":logits, "batch":batch}
    def visualize_batch(computed):
        computed = ptutil.ths2nps(computed)
        batch = computed["batch"]
        Ytg = computed["logits"]
        Ytg = batch["Ytg"]
        vert, face = geoutil.array2mesh(Ytg, thresh=.5)
        img = fresnelvis.renderMeshCloud({"vert":vert,"face":face})
        return {"recon":img}
    def on_sanity_check_end(self, trainer, pl_module):
        print(f"\n{self.__class__.__name__} callback")
        if self.single_vis==True and pl_module.global_rank!=0:
            print("single_vis==True, Only run callback on global_rank 0, current rank:", pl_module.global_rank)
            return 
        if self.no_sanity_check:
            print("no_sanity_check is set to True, skipping...")
            return
        self.process(pl_module, trainer.datamodule.visual_dataloader(), visual_summary=True)
        self.log_summary_images(trainer, pl_module, self.summary_imgs)
    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule, *args, **kwargs) -> None:
        if trainer.current_epoch % self.every_n_epoch == self.every_n_epoch-1:
            print(f"\n{self.__class__.__name__} callback")
            if self.single_vis==True and pl_module.global_rank!=0:
                print("single_vis==True, Only run callback on global_rank 0, current rank:", pl_module.global_rank)
                return 
            try:
                self.process(pl_module, trainer.datamodule.visual_dataloader(), visual_summary=True)
                self.log_summary_images(trainer, pl_module, self.summary_imgs)
            except Exception as err:
                traceback.print_tb(err.__traceback__)
                print("Something is wrong in the callback, skipping...")
    def on_test_start(self, trainer, pl_module, **kwargs):
        self.process_all(pl_module, trainer.datamodule.visual_dataloader(), **kwargs)
    # def on_test_end(self, trainer, pl_module, **kwargs):
    #     self.process_all(pl_module, pl_module.visual_dataloader(), **kwargs)

    def post_training_process(self, trainer, pl_module, data_module, **kwargs):
        if self.visall_after_training_end==True:
            self.process_all(pl_module, data_module.visual_dataloader(), **kwargs)
    def get_summary_imgs(self, imgs, zoomfac=.5):
        all_images = []
        rows = len(imgs)
        for name, image_array in imgs:
            for img_name in image_array:
                img = image_array[img_name]
                all_images.append(img)
        summary = visutil.imageGrid(all_images, shape=(rows, -1), zoomfac=zoomfac)
        return {self.classname: {"caption":self.classname, "image":summary}}
    def log_summary_images(self, trainer, pl_module, summary_imgs, x_axis="epoch"):
        # wandb logger
        import wandb
        for key in summary_imgs:
            t = summary_imgs[key]
            title   = key
            caption = t["caption"]
            image   = t["image"]
            #log_image(trainer, title, caption, image, trainer.global_step)
            #x_val = trainer.current_epoch if x_axis=="epoch" else trainer.global_step
            trainer.logger.experiment.log( \
                {title:[wandb.Image(image,caption=caption)], \
                    "epoch": trainer.current_epoch,
                    "global_step": trainer.global_step})


def null_logger(*args, **kwargs):
    return None
def get_debug_model(trainer, resume=False):
    if resume == True:
        pl_model, train_dloader, val_dloader, test_dloader = trainer.test_mode()
    else:
        pl_model, train_dloader, val_dloader, test_dloader = trainer.test_mode(resume_from=None)
    train_dloader.num_workers = 0 # it will be very slow to invoke subprocesses (num_workers>0)
    val_dloader.num_workers = 0 # it will be very slow to invoke subprocesses (num_workers>0)
    test_dloader.num_workers  = 0
    return pl_model, train_dloader, val_dloader, test_dloader
def debug_model(trainer, resume=False, load_compute=False, load_visual=False, skip_batch_test=False):
    pl_model, train_dloader, val_dloader, test_dloader = get_debug_model(trainer, resume=resume)
    print("Test run train/val step")
    if skip_batch_test==False:
        test_batch(pl_model, train_dloader, val_dloader, test_dloader)

    visual_dloader = trainer.data_module.visual_dataloader()
    for callback in trainer.callbacks:
        cb_name = callback.__class__.__name__
        if cb_name in ["ModelCheckpoint", "ProgressBar", \
            "EarlyStopping", "LearningRateMonitor"]:
            continue
        print("Start callback: ", cb_name)
        returns = callback.process(pl_model, visual_dloader, visual_summary=False, load_compute=load_compute, load_visual=load_visual)
    print("Success")
    return pl_model, train_dloader, val_dloader, test_dloader
def test_batch(pl_model, train_dloader, val_dloader, test_dloader):
    try:
        th_train_batch = ptutil.ths2device(next(iter(train_dloader)), "cuda")
        th_val_batch  = ptutil.ths2device(next(iter(val_dloader)), "cuda")
        origin_logger = pl_model.log
        pl_model.log = null_logger
        #optimizers, schedulers = pl_model.configure_optimizers()
        loss = pl_model.training_step(th_train_batch, batch_idx=0).detach().item()
        print(f"Batch {0} train loss:", loss)
        #for optimizer in optimizers:
        #    print("Testing optimization step of optimizer", optimizer )
        #    optimizer.step()
        loss = pl_model.validation_step(th_val_batch, batch_idx=0).detach().item()
        print(f"Batch {0} val loss:",   loss)
    except Exception as e:
        traceback.print_exc()
        print(e)
        print("failed to run training/validation batch")
        command = input("Press Enter to continue...")
        if command == "stop":
            exit()
    finally:
        pl_model.log = origin_logger
