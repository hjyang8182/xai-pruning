from typing import Literal
import sys
sys.path.append('..') # append parent directory to import

from typing import Optional
from os import path, makedirs
from matplotlib import pyplot as plt
import numpy as np
from math import ceil
from matplotlib import font_manager
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.axes_grid1 import ImageGrid
import json
from torchvision.datasets import ImageFolder
from torch.utils.data import Subset
from ucbm import UCBM
import torch
import seaborn as sns
import pandas as pd
from random import randrange
import math
from tqdm import trange, tqdm


def indices_tensor(targets, class_id: int, equal: bool):
    if equal:
        return (torch.Tensor(targets) == class_id).nonzero().reshape(-1)
    else:
        return (torch.Tensor(targets) != class_id).nonzero().reshape(-1)


class Plotter: 
    '''
    Class that is implementing all plotting tasks for CRAFT-CBMs. 

    Parameter
    ---------
    dpath: str
        The path all results should be saved. 
    '''

    def __init__(self, dpath: str):
        # Check if the given path exists. 
        makedirs(dpath, exist_ok=True)

        result_path = path.join(dpath, "results")
        makedirs(result_path, exist_ok=True)
        
        concept_data_path = path.join(dpath, "concept_data")
        makedirs(concept_data_path, exist_ok=True)

        concept_bank_path = path.join(dpath, "concept_banks")
        makedirs(concept_bank_path, exist_ok=True)
        
        classifier_path = path.join(dpath, "classifier")
        makedirs(classifier_path, exist_ok=True)
        
        self._result_path = result_path
        self._concept_data_path = concept_data_path
        self._concept_bank_path = concept_bank_path
        self._classifier_path = classifier_path

        font_dir = ['plotter/']
        for font in font_manager.findSystemFonts(font_dir):
            font_manager.fontManager.addfont(font)
    
    def get_result_path(self) -> str:
        return self._result_path
    
    def get_concept_data_path(self) -> str:
        return self._concept_data_path
    
    def get_concept_bank_path(self) -> str:
        return self._concept_bank_path
    
    def get_classifier_path(self) -> str:
        return self._classifier_path
    
    def print_concepts_to_file(self, concept_discovery, crops, crops_u, imp, 
                               class_name: str, concepts_to_print: int = -1, 
                               start_concept = 0, print_importances = False):
        '''
        Print the concepts including their importances to a file. 

        Parameter
        ---------
        concept_discovery: CRAFTConceptDiscovery
            Model which was used to compute concepts. 
        crops: np.ndarray
            Image crops the concepts were learned from. 
        crops_u: np.ndarray
            The image crops in the concept basis. 
        imp: list[float]
            The importances of each concept for the class. 
        all_classes: list[int]
            The data classes the concepts were calculated from. 
        class_name: str
            The name used to save the images. 
        concepts_to_print: int, optional
            Amount of concepts that are going to be print in a file. 
        start_concept: int, optional
            Index to start the concept enumeration with. 
        print_importances: bool, optional
            If True, the importances of the given vectors for the current
            class will be printed to a file. 
        '''

        # Load default value (all concepts). 
        if concepts_to_print == -1:
            concepts_to_print = concept_discovery._concepts_per_class
        
        # Check if amount of concepts to print is valid value. 
        if concepts_to_print <= 0 or \
            concepts_to_print > concept_discovery._concepts_per_class:
            raise AttributeError('The parameter concepts_to_print is smaler ' + 
                                 'or equal to zero or larger than the amount ' + 
                                 'concepts calculated by CRAFT. ')
        
        # Plot the importances. 
        if print_importances:
            plt.clf()
            plt.bar(range(len(imp)), imp)
            plt.xticks(range(len(imp)))
            plt.title("Concept Importance")
            plt.savefig(path.join(self._concept_data_path, 
                                  f"Class_{class_name}_Concept_Importance.jpg"))
            plt.close()

        # Sort the concept by importances. 
        most_important_concepts = \
            np.argsort(imp)[::-1][:concepts_to_print]

        # Function to show an image. 
        def show(img, **kwargs):
            img = np.array(img)
            if img.shape[0] == 3:
                img = img.transpose(1, 2, 0)

            img -= img.min();img /= img.max()
            plt.imshow(img, **kwargs); plt.axis('off')
        
        # Print each concept into an image. 
        nb_crops = 9
        for c_id in most_important_concepts:
            best_crops_ids = np.argsort(crops_u[:, c_id])[::-1][:nb_crops]
            best_crops = crops[best_crops_ids]

            plt.clf()
            plt.figure(figsize=(5, 5))
            plt.axis('off')
            # plt.title("Concept " + str(c_id + start_concept))
            # plt.title("Concept " + str(c_id + start_concept) + 
            #           " has an importance value of " + str(importances[c_id]))
            width = int(nb_crops ** 0.5)
            height = ceil(nb_crops / width)
            for i in range(nb_crops):
                plt.subplot(width, height, i+1)
                show(best_crops[i])
            plt.savefig(path.join(self._concept_data_path, 
                                  f"Concept_{(c_id + start_concept):03}_-_Class_{class_name}.jpg"))
            plt.close()

    def print_concepts_to_file_new(self, dataset, crops_u, patch_size: int, n_patches: int, strides: float, save_dir: str, nb_crops: int = 9, nb_crops_json: int = 100, verbose: bool = False):
        # Function to show an image. 
        def show(ax, img, **kwargs):
            img = np.array(img)
            if img.shape[0] == 3:
                img = img.transpose(1, 2, 0)

            img -= img.min();img /= img.max()
            ax.imshow(img, **kwargs); ax.axis('off')
        
        # Print each concept into an image. 
        assert nb_crops > 0
        sqrt_nb_crops = int(math.sqrt(nb_crops))
        iterator = trange(crops_u.shape[1], leave=False) if verbose else range(crops_u.shape[1])
        fill = ceil(math.log10(crops_u.shape[1]))
        crops_ids_res = {}
        for c_id in iterator:
            fig = plt.figure()
            grid = ImageGrid(fig, 111, nrows_ncols=(ceil(nb_crops/sqrt_nb_crops), sqrt_nb_crops), axes_pad=0.1)

            argsorted_crops_ids = np.argsort(crops_u[:, c_id])[::-1]
            crops_ids_res[c_id] = list(map(int, argsorted_crops_ids[:min(nb_crops_json, len(argsorted_crops_ids))]))
            best_crops_ids = argsorted_crops_ids[:nb_crops]
            img_ids = best_crops_ids // n_patches
            patch_ids = best_crops_ids % n_patches + np.array(list(range(nb_crops)))*n_patches
            all_images = torch.stack([dataset[img_id][0] for img_id in img_ids], dim=0)
            patches = torch.nn.functional.unfold(all_images, kernel_size=patch_size, stride=strides)
            patches = patches.transpose(1, 2).contiguous().view(-1, 3, patch_size, patch_size)
            best_crops = patches[patch_ids]

            for ax, i in zip(grid, range(nb_crops)):
                show(ax, best_crops[i])
            
            plt.savefig(path.join(save_dir, f"Concept_{str(c_id).zfill(fill)}.jpg"))
            plt.close()

        with open(path.join(save_dir, "crops_ids.json"), "w") as f:
            json.dump(crops_ids_res, f, indent=2)
    
    def plot_images_with_concept_similarities_to_file(
            self, pictures: np.ndarray, similarities: np.ndarray, 
            names: Optional[list[str]] = None, concepts_to_show: int = 10, 
            concept_labels: Optional[list[str]] = None):
        '''
        Plot images with the correponding similarities to file. 

        Parameter
        ---------
        pictures: np.ndarray
            Pictures in shape (N, C, W, H). 
        similarities: np.ndarray
            Similarities in shape (N, #concepts)
        names: list[str], optional
            Names corresponding to pictures. 
        concepts_to_show: int, optional
            Amount of best concepts to show. 
        concept_labels: list[str], optional
            A list of the concept labels. 
        '''
        
        # Check given inputs. 
        assert names is None or len(names) == pictures.shape[0], \
            'Name list and amount of pictures has to be equal. '
        assert similarities.shape[0] == pictures.shape[0], \
            'Amount of pictures in pictures and similarities must be equal. '
        
        if names is None:
            names = [f'Image_{i}' for i in range(pictures.shape[0])]

        # Function to show an image. 
        def show(img, ax, **kwargs):
            img = np.array(img)
            if img.shape[0] == 3:
                img = img.transpose(1, 2, 0)

            img -= img.min();img /= img.max()
            ax.imshow(img, **kwargs); ax.axis('off')

        # print images. 
        n = len(names)
        for i in range(n):
            # Set up plot. 
            plt.clf()
            plt.rcParams.update({'font.size': 16})
            plt.rcParams.update({'font.family': 'CMU Serif'})
            fig, ax = plt.subplots(1, 3, width_ratios=[1, 2, 2])
            fig.set_figwidth(12)
            fig.set_figheight(6)
            # fig.suptitle(names[i])
            ax[1].axis('off')

            # Extract simularites for image i and sort them by simularity. 
            sim = similarities[i,:]
            sim_sorted = np.sort(sim)[::-1]
            sim_argsorted = np.argsort(sim)[::-1]

            # Show the image itself. 
            show(pictures[i,:,:,:], ax[0])
            
            # Plot the bars shwoing the concept simularities. 
            y_pos = np.arange(concepts_to_show)
            labels = [(f'Concept {i}' 
                      if concept_labels is None else 
                      f'{concept_labels[i]} ({i})')# + f': {sim[i]:.2f}%'
                      for i in sim_argsorted[:concepts_to_show]]
            bar_con = ax[-1].barh(y_pos, sim_sorted[:concepts_to_show], 
                                 color="#00376d", align="center")
            ax[-1].bar_label(bar_con, 
                             labels=[str(round(sim, 4)) 
                                     for sim in sim_sorted[:concepts_to_show]], 
                             label_type='center', color='white')
            ax[-1].set_yticks(y_pos, labels=labels)
            ax[-1].set_xlabel('concept simularity')
            ax[-1].invert_yaxis()

            # Save the plot to file. 
            plt.savefig(path.join(self._result_path, 
                                  f"Concept_Sim_{names[i]}.jpg"), dpi=300)
            plt.close()
    
    def plot_concept_violin_plot(self, 
                                 cons: list[int], 
                                 class_id: int, 
                                 class_label: str, 
                                 similarities: np.ndarray, 
                                 targets: np.ndarray, 
                                 concept_labels: Optional[list[str]] = None,
                                 importances: Optional[list[str]] = None,
                                 tmp: str = ""):
        '''
        Plot violin plots of the given concept numbers for the class class_id. 

        Parameters
        ----------
        cons: list[int]
            List of the concept numbers the violin plot should be done for. 
        class_id: int
            The class each concept is corresponding to. 
        class_label: str
        similarities: np.ndarray
            The concept similarities of n images in a matrix of shape 
            (n, #concepts). 
        targets: np.ndarray
            The targets of the images in a vector of shape (n). 
        concept_labels = Optional[list[str]] = None
            The concept labels. 
        importances: Optional[list[str]] = None
            The importance score of each concept
        tmp: str = ""
            String that is also inputted into the file name. 
        '''

        similarities = torch.stack(
            [similarities[i] for i in range(len(similarities))])

        # Function that returns the indexes of a sequence, that are label
        # with the given class_id. 
        def indices_tensor(targets, class_id, eq=True):
            if eq:
                return (torch.Tensor(targets) == class_id).nonzero().reshape(-1)
            else:
                return (torch.Tensor(targets) != class_id).nonzero().reshape(-1)

        sim_class = dict()
        sim_non_class = dict()
        for con in cons:
            sim_class[con] = \
                similarities[indices_tensor(targets, class_id), con].tolist()
            sim_non_class[con] = \
                similarities[indices_tensor(targets, class_id, eq=False), 
                             con].tolist()
        
        imp_dict = {con: "" for con in cons}
        if importances is not None:
            imp_dict = {con: f"{importances[i]:.4f}" 
                        for i, con in enumerate(cons)}

        df = pd.DataFrame({
            "concept simularity": sum((sim_class[con] + sim_non_class[con] 
                                      for con in cons), []), 
            "belonging": sum((["class data"] * len(sim_class[con]) + 
                                ["non data class"] * len(sim_non_class[con]) 
                                for con in cons), []), 
            "concept": sum(([
                (f"Concept {con}" 
                if concept_labels is None 
                else f"{concept_labels[con]} ({con})") + f" {imp_dict[con]}"
            ] * (len(sim_class[con]) + len(sim_non_class[con])) 
                           for con in cons), [])})
        
        plt.clf()
        fig, ax = plt.subplots(1, 2)
        ax[0].axis('off')
        sns.violinplot(data=df, x="concept simularity", y="concept", hue="belonging", 
                       split=True, inner="quart", cut=0, ax=ax[-1])
        
        fig.suptitle(f"class {class_label} - concept similarity")
        plt.savefig(path.join(self._result_path, 
                              f"class_{class_id}_{tmp}_concept_violin_plot.jpg"), 
                    dpi=300)
        plt.close()

    def plot_classifier_weights(self, 
                                coeff: np.ndarray, 
                                best_coeff_to_plot: int, 
                                class_to_idx: dict[str, int], 
                                concept_labels: Optional[list[str]] = None, 
                                tmp: str = ""):
        '''
        Plot a blot bar which shows the coeff of the linear classifier for the
        given class. 

        Parameters
        ----------
        coeff: np.ndarray
            The coefficients for the each class for every concept. 
            Shape (#concepts, #classes). 
        best_coeff_to_plot: int
            That much best coefficients can be seen as a single bar in the plot. 
            The others are combined to rest. 
        class_to_idx: dict[str, int]
            Provides a dictionary with all class names and class ids
        concept_labels: Optional[list[str]] = None
            The labels of each concept. 
        tmp: str = ""
            String that is also inputted into the file name. 
        '''

        for cls_name, cls_id in class_to_idx.items():
            abs_coeff = np.abs(coeff[:,cls_id])
            co_argsorted = np.argsort(abs_coeff)[::-1]
            co_argsorted_used = co_argsorted[:best_coeff_to_plot]
            co_argsorted_unused = co_argsorted[best_coeff_to_plot:]
            co_sorted_used = abs_coeff[co_argsorted_used]
            co_inv = coeff[:,cls_id] < 0
            co_sorted_unused = abs_coeff[co_argsorted_unused]
            co_sorted = np.append(co_sorted_used, np.sum(co_sorted_unused))
            co_argsorted = [f"{'NOT ' if co_inv[i] else ''}Concept {i}" 
                            if concept_labels is None else 
                            f"{'NOT ' if co_inv[i] else ''}{concept_labels[i]} ({i})"
                            for i in co_argsorted_used] + ["others"]
        
            plt.clf()
            fig, ax = plt.subplots(1, 2)
            ax[0].axis('off')
            y_pos = np.arange(best_coeff_to_plot+1)
            bar_con = ax[-1].barh(y_pos, co_sorted, color="#00376d", align="center")
            ax[-1].bar_label(bar_con, 
                                labels=[str(round(sim, 4)) 
                                        for sim in co_sorted], 
                                label_type='center', color='white')
            ax[-1].set_yticks(y_pos, labels=co_argsorted)
            ax[-1].set_xlabel('weight')
            ax[-1].invert_yaxis()
            fig.suptitle(f"Classifier weights of class {cls_name}")
            plt.savefig(path.join(self._result_path, 
                                f"class_{cls_id}_{tmp}_weights.jpg"), 
                        dpi=300)
            plt.close()
    
    def plot_gate_distribution(self,
                               ph_cbm: UCBM,
                               threshold: float = 0.5,
                               tmp: str = "",
                               save_dir: Optional[str] = None):
        '''
        Plot a histogram of the learned gate values (sigmoid(gate_logits))
        of a gated UCBM classifier, marking the open/closed threshold and
        the resulting amount of open concepts.

        Parameters
        ----------
        ph_cbm: UCBM
            A UCBM instance fit with gated=True.
        threshold: float = 0.5
            Gate value above which a concept counts as open.
        tmp: str = ""
            String that is also inputted into the file name.
        save_dir: Optional[str] = None
            Directory to save the plot into. Defaults to this plotter's shared
            results directory; pass a run's own class_path to save the plot
            alongside that run's classifier.pth/info.json instead.
        '''

        gates = ph_cbm.get_gate_values().cpu().numpy()
        n_open = int((gates > threshold).sum())

        plt.clf()
        plt.rcParams.update({'font.size': 16})
        plt.rcParams.update({'font.family': 'CMU Serif'})
        fig, ax = plt.subplots()
        fig.set_figwidth(8)
        fig.set_figheight(6)

        ax.hist(gates, bins=50, range=(0, 1), color="#00376d")
        ax.axvline(threshold, color="red", linestyle="--",
                   label=f"threshold ({threshold})")
        ax.set_xlabel("gate value")
        ax.set_ylabel("number of concepts")
        ax.legend()
        fig.suptitle(f"Gate distribution - {n_open}/{len(gates)} concepts open")

        plt.savefig(path.join(save_dir or self._result_path,
                              f"{tmp}_gate_distribution.jpg"),
                    dpi=300)
        plt.close()

    def plot_lambda_gate_sweep(self,
                               lambda_gates: list[float],
                               accs: list[float],
                               n_open: list[float],
                               dataset_name: str,
                               backbone_name: str,
                               accs_std: Optional[list[float]] = None,
                               n_open_std: Optional[list[float]] = None,
                               baseline_acc: Optional[float] = None,
                               baseline_acc_std: Optional[float] = None,
                               metric_name: str = "Accuracy",
                               split_label: str = "val",
                               save_dir: Optional[str] = None,
                               accs2: Optional[list[float]] = None,
                               accs2_std: Optional[list[float]] = None,
                               baseline_acc2: Optional[float] = None,
                               baseline_acc2_std: Optional[float] = None,
                               metric2_name: Optional[str] = None):
        '''
        Twin-axis plot of a metric (accuracy or CEA) and open-gate count
        against lambda_gate for a sweep of gated UCBM runs, with the
        ungated baseline drawn as a horizontal reference line. Optionally a
        second metric (accs2) is overlaid on the same left axis so both
        curves -- e.g. CEA and raw accuracy -- can be read against each
        other in one figure.

        Parameters
        ----------
        lambda_gates: list[float]
            x-axis values, one gated run (mean over seeds) per entry.
        accs: list[float]
            Metric value (mean over seeds) for each lambda_gate.
        n_open: list[float]
            Mean open-gate count for each lambda_gate.
        dataset_name: str
        backbone_name: str
        accs_std, n_open_std: Optional[list[float]] = None
            Std over seeds, shown as a shaded band if given.
        baseline_acc, baseline_acc_std: Optional[float] = None
            Ungated baseline's value of the same metric, drawn as a dashed
            reference line (with a shaded std band if given).
        metric_name: str = "Accuracy"
            Name of the left-axis metric, e.g. "Accuracy" or "CEA".
        split_label: str = "val"
            Used only in the saved file name (e.g. "val" vs "test").
        save_dir: Optional[str] = None
            Directory to save the plot into. Defaults to this plotter's
            shared results directory.
        accs2, accs2_std: Optional[list[float]] = None
            Second metric (mean/std over seeds) to overlay on the left axis.
        baseline_acc2, baseline_acc2_std: Optional[float] = None
            Ungated baseline's value of the second metric, drawn as a dashed
            reference line in the second metric's colour.
        metric2_name: Optional[str] = None
            Name of the second metric, e.g. "Accuracy". Required if accs2 is
            given; also used to build the left-axis label "<metric_name> /
            <metric2_name>".
        '''

        ACC_COLOR, GATE_COLOR = "#00376d", "#c0654b"
        ACC2_COLOR = "#2e7d5b"
        metric_label = metric_name if metric_name.isupper() else metric_name.lower()
        has_second = accs2 is not None
        metric2_label = None
        if has_second:
            metric2_label = metric2_name if metric2_name.isupper() else metric2_name.lower()

        plt.clf()
        plt.rcParams.update({'font.size': 14})
        plt.rcParams.update({'font.family': 'CMU Serif'})
        fig, ax1 = plt.subplots(figsize=(7.5, 5))

        if accs_std:
            lo = [a - s for a, s in zip(accs, accs_std)]
            hi = [a + s for a, s in zip(accs, accs_std)]
            ax1.fill_between(lambda_gates, lo, hi, color=ACC_COLOR, alpha=0.15, linewidth=0)
        ax1.plot(lambda_gates, accs, color=ACC_COLOR, marker='o', markersize=6,
                 markeredgecolor='white', linewidth=2.25, label=f'Gated {metric_label}')
        if baseline_acc is not None:
            if baseline_acc_std:
                ax1.axhspan(baseline_acc - baseline_acc_std, baseline_acc + baseline_acc_std,
                            color=ACC_COLOR, alpha=0.08)
            ax1.axhline(y=baseline_acc, color=ACC_COLOR, linestyle='--', linewidth=1.5,
                        label=f'Baseline {metric_label}')

        if has_second:
            if accs2_std:
                lo = [a - s for a, s in zip(accs2, accs2_std)]
                hi = [a + s for a, s in zip(accs2, accs2_std)]
                ax1.fill_between(lambda_gates, lo, hi, color=ACC2_COLOR, alpha=0.15, linewidth=0)
            ax1.plot(lambda_gates, accs2, color=ACC2_COLOR, marker='^', markersize=6,
                     markeredgecolor='white', linewidth=2.25, label=f'Gated {metric2_label}')
            if baseline_acc2 is not None:
                if baseline_acc2_std:
                    ax1.axhspan(baseline_acc2 - baseline_acc2_std, baseline_acc2 + baseline_acc2_std,
                                color=ACC2_COLOR, alpha=0.08)
                ax1.axhline(y=baseline_acc2, color=ACC2_COLOR, linestyle='--', linewidth=1.5,
                            label=f'Baseline {metric2_label}')

        ax1.set_xscale('log')
        # Plain (non-mathtext) tick labels -- the default log-scale formatter
        # renders exponents via mathtext, whose minus-sign glyph isn't present
        # in every fontset and silently degrades to a placeholder symbol.
        ax1.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:g}'))
        ax1.set_xlabel(r'$\lambda_{gate}$')
        if has_second:
            ax1.set_ylabel(f"{metric_name} / {metric2_name}", color='black')
            ax1.tick_params(axis='y', labelcolor='black')
        else:
            ax1.set_ylabel(metric_name, color=ACC_COLOR)
            ax1.tick_params(axis='y', labelcolor=ACC_COLOR)

        ax2 = ax1.twinx()
        if n_open_std:
            lo = [max(0, o - s) for o, s in zip(n_open, n_open_std)]
            hi = [o + s for o, s in zip(n_open, n_open_std)]
            ax2.fill_between(lambda_gates, lo, hi, color=GATE_COLOR, alpha=0.12, linewidth=0)
        ax2.plot(lambda_gates, n_open, color=GATE_COLOR, marker='D', markersize=5,
                 markeredgecolor='white', linewidth=1.75, linestyle='--', label='Open gates')
        ax2.set_ylabel('Open gates', color=GATE_COLOR)
        ax2.tick_params(axis='y', labelcolor=GATE_COLOR)
        ax2.grid(False)

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=10.5,
                   loc='upper center', bbox_to_anchor=(0.5, -0.15),
                   ncol=3 if not has_second else 2, frameon=False)
        fig.suptitle(f"{dataset_name}-{backbone_name}: lambda_gate sweep ({split_label})")

        fig.tight_layout()
        plt.savefig(path.join(save_dir or self._result_path,
                              f"lambda_gate_sweep_{split_label}.jpg"),
                    dpi=300, bbox_inches='tight')
        plt.close()

    def plot_pareto_frontier(self,
                             open_gates: list[float],
                             accs: list[float],
                             lambda_gates: list[float],
                             dataset_name: str,
                             backbone_name: str,
                             open_gates_std: Optional[list[float]] = None,
                             accs_std: Optional[list[float]] = None,
                             baseline_open_gates: Optional[float] = None,
                             baseline_acc: Optional[float] = None,
                             baseline_acc_std: Optional[float] = None,
                             metric_name: str = "Accuracy",
                             split_label: str = "val",
                             save_dir: Optional[str] = None):
        '''
        Scatter accuracy directly against open-gate count, one point per
        lambda_gate, each labeled with its lambda_gate value. Unlike CEA
        (which collapses both into one scalar via a beta-weighted ratio),
        this makes the raw accuracy/sparsity trade-off visible so a "these
        all tie on accuracy" situation -- where CEA would silently just
        reward whichever run is sparsest -- is obvious by eye instead:
        points at the same height mean the metric has no signal there,
        no matter what CEA says.

        Parameters
        ----------
        open_gates: list[float]
            x-axis values (mean open-gate count over seeds), one per
            lambda_gate.
        accs: list[float]
            y-axis values (mean accuracy over seeds), one per lambda_gate.
        lambda_gates: list[float]
            Used only to label each point.
        dataset_name: str
        backbone_name: str
        open_gates_std, accs_std: Optional[list[float]] = None
            Std over seeds, drawn as error bars if given.
        baseline_open_gates, baseline_acc, baseline_acc_std: Optional[float] = None
            Ungated baseline's point, drawn separately so it's clearly not
            part of the lambda_gate curve.
        metric_name: str = "Accuracy"
            Name of the y-axis metric.
        split_label: str = "val"
            Used only in the saved file name and title (e.g. "val" vs "test").
        save_dir: Optional[str] = None
            Directory to save the plot into. Defaults to this plotter's
            shared results directory.
        '''

        POINT_COLOR, BASELINE_COLOR = "#00376d", "#c0654b"

        plt.clf()
        plt.rcParams.update({'font.size': 14})
        plt.rcParams.update({'font.family': 'CMU Serif'})
        fig, ax = plt.subplots(figsize=(7.5, 5))

        order = sorted(range(len(open_gates)), key=lambda i: open_gates[i])
        xs = [open_gates[i] for i in order]
        ys = [accs[i] for i in order]
        lgs = [lambda_gates[i] for i in order]

        xerr = [open_gates_std[i] for i in order] if open_gates_std else None
        yerr = [accs_std[i] for i in order] if accs_std else None
        ax.errorbar(xs, ys, xerr=xerr, yerr=yerr, color=POINT_COLOR,
                    marker='o', markersize=7, markeredgecolor='white',
                    linewidth=2.0, capsize=3, label=r'Gated ($\lambda_{gate}$ sweep)')
        for x, y, lg in zip(xs, ys, lgs):
            ax.annotate(f'{lg:g}', (x, y), textcoords="offset points",
                        xytext=(0, 9), ha='center', fontsize=9.5, color=POINT_COLOR)

        if baseline_acc is not None and baseline_open_gates is not None:
            ax.errorbar([baseline_open_gates], [baseline_acc],
                        yerr=[baseline_acc_std] if baseline_acc_std else None,
                        color=BASELINE_COLOR, marker='*', markersize=14,
                        markeredgecolor='white', capsize=3, label='Baseline', linestyle='none')

        ax.set_xlabel('Open gates')
        ax.set_ylabel(metric_name)
        ax.legend(fontsize=10.5, loc='upper center', bbox_to_anchor=(0.5, -0.15),
                  ncol=2, frameon=False)
        fig.suptitle(f"{dataset_name}-{backbone_name}: accuracy vs. open gates ({split_label})")

        fig.tight_layout()
        plt.savefig(path.join(save_dir or self._result_path,
                              f"pareto_frontier_{split_label}.jpg"),
                    dpi=300, bbox_inches='tight')
        plt.close()

    @torch.no_grad()
    def plot_example_pictures(self,
                              dataset: ImageFolder, 
                              ph_cbm: UCBM, 
                              img_per_class: int, 
                              concepts_to_show: int, 
                              tmp: str = ""):
        
        # Function to show an image. 
        def show(img, ax, **kwargs):
            img = np.array(img)
            if img.shape[0] == 3:
                img = img.transpose(1, 2, 0)

            img -= img.min();img /= img.max()
            ax.imshow(img, **kwargs); ax.axis('off')
        
        concept_labels = ph_cbm._concept_labels
        weights = ph_cbm.get_classifier_weights().cpu()
        for j, cls_id in tqdm(enumerate(dataset.class_to_idx.values()), leave=False, total=len(dataset.class_to_idx)):
            indices = []
            idx_to_cls = {v: k for k, v in dataset.class_to_idx.items()}
            for _ in trange(img_per_class, leave=False):
                r = randrange(len(dataset))
                while dataset.targets[r] != cls_id or r in indices:
                    r = randrange(len(dataset))
                indices.append(r)
        
            subset = Subset(dataset, indices)
            subset = torch.stack([subset[i][0] for i in range(len(subset))], dim=0)

            preds, con_acts = ph_cbm.predict(subset)
            for i in range(subset.shape[0]):
                con_acts[i,:] = con_acts[i,:] * weights[torch.argmax(preds[i]).item(),:]
        
            for i in range(len(indices)):
                plt.clf()
                pred = preds[i]
                con_act = con_acts[i,:]
                con_act_asorted = torch.argsort(torch.abs(con_act), descending=True)
                con_act_sorted = torch.abs(con_act[con_act_asorted])

                plt.rcParams.update({'font.size': 16})
                plt.rcParams.update({'font.family': 'CMU Serif'})
                fig, ax = plt.subplots(1, 3, width_ratios=[1, 1, 2])
                fig.set_figwidth(12)
                fig.set_figheight(6)
                pred_s = torch.argmax(pred).item()
                fac = torch.nn.functional.softmax(pred, dim=0)[pred_s].item()
                fig.suptitle(f"{idx_to_cls[cls_id]}; \n"
                            f"Pred: {idx_to_cls[pred_s]}, {100*fac:.2f}%")
                ax[1].axis('off')

                # Show the image itself. 
                show(subset[i,:,:,:], ax[0])
                
                # Plot the bars shwoing the concept simularities. 
                y_pos = np.arange(concepts_to_show+1)
                labels = [(("NOT " if con_act[j] < 0 else "") + 
                        (f'Concept {j}' 
                        if concept_labels is None else 
                        f'{concept_labels[j]} ({j})'))
                        for j in con_act_asorted[:concepts_to_show]] + \
                            [f"{con_act_asorted.numel() - concepts_to_show} others"]
                cons = torch.cat((con_act_sorted[:concepts_to_show], 
                                torch.sum(con_act_sorted[concepts_to_show:]).unsqueeze(0)), 
                                dim=0)

                bar_con = ax[-1].barh(y_pos, cons, color="#00376d", align="center")
                ax[-1].bar_label(bar_con, 
                                labels=[str(round(float(sim.item()), 4)) for sim in cons], 
                                label_type='center', color='white')
                ax[-1].set_yticks(y_pos, labels=labels)
                ax[-1].set_xlabel('concept simularity * class-concept weight')
                ax[-1].invert_yaxis()

                # Save the plot to file. 
                plt.savefig(path.join(self._result_path, 
                                    f"Example_{tmp}_{i+j*img_per_class}_{idx_to_cls[cls_id]}.jpg"), dpi=300)
                plt.close()