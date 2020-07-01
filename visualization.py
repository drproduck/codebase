import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
from torchvision.utils import make_grid
import numpy as np
import json
import imageio
from util.general import get_all_files
import torch
from PIL import Image


def view_points(points_list, points_labels, points_titles,
                iteration, prefix_exp, figsize=5, save=True, view=False):
    num_points_plot = len(points_list)
    fig, axs = plt.subplots(1, num_points_plot, figsize=(figsize * num_points_plot, figsize), squeeze=False)

    # PLOT POINTS
    for i in range(num_points_plot):
        points = points_list[i]
        labels = points_labels[i]
        name = points_titles[i]
        dim = points.shape[-1]              

        if dim == 2:
            axs[0][i].scatter(points[:, 0], points[:, 1], c=labels, cmap=plt.cm.Spectral)
            axs[0][i].set_title(name)
        elif dim == 3:
            axs[0][i] = fig.add_subplot(1, num_points_plot, i + 1, projection='3d')
            axs[0][i].scatter(points[:, 0], points[:, 1], points[:, 2], c=labels, cmap=plt.cm.Spectral)
            axs[0][i].view_init(elev=20, azim=-80)
            axs[0][i].set_title(name)
        elif dim == 1:
            axs[0][i].scatter(points[:, 0], 0, c=labels, cmap=plt.cm.Spectral)
            axs[0][i].set_title(name)
        else:            
            raise NotImplementedError

    if save:
        plt.savefig(f"{prefix_exp}/output_{'{:08d}'.format(iteration)}.png", bbox_inches='tight')

    if view:
        plt.show()

    plt.close()


def view_images(outputs, titles=None, num_images=16, channels=(), image_sizes=(), nrow=4, figsize=10, save_path=None,
                view=False, fig_title=None):
    num_fig = len(outputs)
    fig, axs = plt.subplots(1, num_fig, figsize=(figsize, 4 * num_fig), squeeze=False)

    for i in range(num_fig):
        image = outputs[i][:num_images, :].view(num_images, channels[i], image_sizes[i], image_sizes[i]).cpu().detach()
        image = make_grid(image, nrow=nrow)
        image = image.permute(1, 2, 0)

        axs[0][i].set_xticks([])
        axs[0][i].set_yticks([])
        axs[0][i].imshow(image)

        if titles is not None:
            axs[0][i].set_title(titles[i])

    if fig_title is not None:
        fig.suptitle(fig_title)

    if view:
        plt.show()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def view_points_and_plots(points_list, points_labels, points_titles,
                          losses, loss_titles, logger, iterations,
                          prefix_exp, figsize=5):
    num_points_plot = len(points_list)
    num_loss_plot = len(losses)
    max_row = max(num_points_plot, num_loss_plot)
    fig, axs = plt.subplots(2, max_row, figsize=(figsize * max_row, figsize * 2), squeeze=False)

    # PLOT POINTS
    for i in range(len(points_list)):
        points = points_list[i]
        labels = points_labels[i]
        name = points_titles[i]
        dim = points.shape[-1]

        if dim == 2:
            axs[0][i].scatter(points[:, 0], points[:, 1], c=labels, cmap=plt.cm.Spectral)
            axs[0][i].set_title(name)
        elif dim == 3:
            axs[0][i] = fig.add_subplot(1, num_fig, i + 1, projection='3d')
            axs[0][i].scatter(points[:, 0], points[:, 1], points[:, 2], c=labels, cmap=plt.cm.Spectral)
            axs[0][i].view_init(elev=20, azim=-80)
            axs[0][i].set_title(name)
        elif dim == 1:
            axs[0][i].scatter(points[:, 0], np.zeros_like(points[:, 0]), c=labels, cmap=plt.cm.Spectral)
            axs[0][i].set_title(name)
        else:
            raise NotImplementedError
            
    # PLOT LOSSES
    for i in range(len(losses)):
        axs[1][i].plot(iterations, losses[i],
                                         color="#66892B", marker="o", markersize=2, linewidth=1,
                                         label=loss_titles[i])

        axs[1][i].set_xlim(-1, iterations[-1] + 1)
        axs[1][i].set_title(loss_titles[i])
        

    plt.savefig(f"{prefix_exp}/output_{'{:08d}'.format(iterations[-1])}.png", bbox_inches='tight')
    plt.close()

    with open(f"{prefix_exp}/log.json", 'w') as file:
        json.dump(logger, file)


def view_images_and_losses(num_grid, images, image_titles, image_sizes, image_channels,
                           losses, loss_titles, logger, iterations,
                           prefix_exp, nrow=4, figsize=5):
    num_image_plot = len(images)
    num_loss_plot = len(losses)
    num_fig = num_image_plot + num_loss_plot

    fig, axs = plt.subplots(1, num_fig, figsize=(figsize * num_fig, figsize), squeeze=False)

    # PLOT IMAGES
    for i in range(num_image_plot):
        image = images[i][:num_grid, :].view(num_grid, image_channels[i], image_sizes[i], image_sizes[i])
        image = make_grid(image, nrow=nrow)
        image = image.permute(1, 2, 0)

        axs[0][i].set_xticks([])
        axs[0][i].set_yticks([])
        axs[0][i].imshow(image)
        axs[0][i].set_title(image_titles[i])

    # PLOT LOSSES
    for i in range(num_loss_plot):
        axs[0][i + num_image_plot].plot(iterations, losses[i],
                                        color="#66892B", linewidth=1,
                                        label=loss_titles[i])

        axs[0][i + num_image_plot].set_xlim(-1, iterations[-1] + 1)
        axs[0][i + num_image_plot].set_title(loss_titles[i])

    plt.savefig(f"{prefix_exp}/output_{'{:08d}'.format(iterations[-1])}.png", bbox_inches='tight')
    plt.close()

    with open(f"{prefix_exp}/log.json", 'w') as file:
        json.dump(logger, file)


def to_gif(frame_dir, save_path="out.gif", fps=12, skip=1):
    images = []
    file_names = get_all_files(frame_dir, keep_dir=True, sort=True)
    file_names = file_names[1:]

    num_file = len(file_names)

    for i in range(num_file):
        if i % skip != 0 and i != num_file - 1:
            continue

        filename = file_names[i]

        image = imageio.imread(filename)[:, :, :3]
        H, W, C = image.shape
        Hb = 10

        percentage = int((1.0 * (i + 1) * W) / num_file)

        progress_bar = np.concatenate([
            np.ones((Hb, W, 1)) * 255,
            np.zeros((Hb, W, 1)),
            np.zeros((Hb, W, 1))
        ], axis=-1)

        progress_bar[:, percentage:, :] = 255

        progress_bar = progress_bar.astype("uint8")

        image = np.concatenate([
            image,
            progress_bar
        ], axis=0)

        images.append(image)

    imageio.mimsave(save_path, images, fps=fps)

    
    
class plt_viz():
    def __init__(self):
        self.scatter_points= []
        self.scatter_labels = []
        self.scatter_titles = []
        
        self.iterations = []
        self.loss_values = []
        self.loss_titles = []
        
        self.img_tensors = []
        self.img_titles = []
        
        self.plots = {'scatter': [], 'loss': [], 'image': []}
        self.fig = None
        self.axs = None
        
    def add(self, plot_type, **kwargs):
        if plot_type == 'scatter':
            self.scatter_points.append(kwargs['points'])
            self.scatter_labels.append(kwargs['labels'])
            self.scatter_titles.append(kwargs['title'])
            return [0, len(self.scatter_points)-1]
        if plot_type == 'loss':
            self.iterations.append(kwargs['x'])
            self.loss_values.append(kwargs['y'])
            self.loss_titles.append(kwargs['title'])
            return [1, len(self.loss_values)-1]
        if plot_type == 'image':
            self.img_tensors.append(kwargs['tensor'])
            self.img_titles.append(kwargs['title'])
            return [2, len(self.img_tensors)-1]

        
        
    def draw(self, figsize=5):
        n_scatter = len(self.scatter_points)
        n_loss = len(self.loss_values)
        n_img = len(self.img_tensors)
        
        max_row = max(n_scatter, n_loss, n_img)
        fig, axs = plt.subplots(3, max_row, figsize=(figsize * max_row, figsize * 3), squeeze=False)
        
        for i in range(n_scatter):
            
            points = self.scatter_points[i]
            labels = self.scatter_labels[i]
            name = self.scatter_titles[i]
            dim = points.shape[-1]
            
            if dim == 2:
                axs[0][i].scatter(points[:, 0], points[:, 1], c=labels, cmap=plt.cm.Spectral)
                axs[0][i].set_title(name)
            elif dim == 3:
                axs[0][i] = fig.add_subplot(3, max_row, i + 1, projection='3d')
                axs[0][i].scatter(points[:, 0], points[:, 1], points[:, 2], c=labels, cmap=plt.cm.Spectral)
                axs[0][i].view_init(elev=20, azim=-80)
                axs[0][i].set_title(name)
            elif dim == 1:
                axs[0][i].scatter(points[:, 0], np.zeros_like(points[:, 0]), c=labels, cmap=plt.cm.Spectral)
                axs[0][i].set_title(name)
            else:
                pass 
#                 raise NotImplementedError
        
        # PLOT LOSSES
        for i in range(n_loss):
            axs[1][i].plot(self.iterations[i], self.loss_values[i],
                           color="#66892B", marker="o", markersize=2, linewidth=1,
                           label=self.loss_titles[i]
                          )

            axs[1][i].set_xlim(-1, self.iterations[i][-1] + 1)
            axs[1][i].set_title(self.loss_titles[i])
            
        # PLOT IMAGES
        for i in range(n_img):
            ndarr = self.img_tensors[i].mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
            im = Image.fromarray(ndarr)
            axs[2][i].imshow(im)
            axs[2][i].set_xticks([])
            axs[2][i].set_yticks([])
            axs[2][i].set_title(self.img_titles[i])
            
        self.fig = fig
        self.axs = axs
        
        return axs

        
    def save(self, prefix_exp, postfix_exp):
        self.fig.savefig(f"{prefix_exp}/output_{postfix_exp}.png", bbox_inches='tight')
        plt.close(self.fig)