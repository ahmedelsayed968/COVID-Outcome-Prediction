from typing import Optional, Union

import numpy as np
from enum import Enum
from sklearn.manifold import TSNE
from umap import UMAP


class PlotType(Enum):
    UMAP = 0
    TSNE = 1


def plot_data(features: np.array,
              target: Union[np.array, None],
              method: PlotType = PlotType.UMAP,
              n_component: Optional[int] = 2):
    match method:
        case PlotType.UMAP:
            umap_md = UMAP(n_components=n_component,
                           init='random',
                           random_state=2023, )
            return umap_md.fit_transform(X=features,
                                             y=target)
        case PlotType.TSNE:
            tsne_md = TSNE(n_components=n_component,
                           learning_rate='auto',
                           random_state=2023)
            return tsne_md.fit_transform(X=features,y=target)

