{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "import tarfile\n",
    "import numpy as np\n",
    "import sys\n",
    "\n",
    "\n",
    "datasets_root = './datasets'\n",
    "from .text import TextDataset\n",
    "from . import util\n",
    "\n",
    "class BHOT():\n",
    "    def __init__(self, data=None):\n",
    "        self.data = data\n",
    "        self.y_onehot=False\n",
    "        self.batch_size = 128\n",
    "        self.batch_shuffle = False\n",
    "        self.init_part()\n",
    "        self.n_classes = 0\n",
    "        \n",
    "        #self.skip_window=1     # [ skip_window target skip_window ]\n",
    "        #self.vocab_len = 0\n",
    "        #self.id2word = {}\n",
    "        \n",
    "        self.dataset_name='bhot'\n",
    "        self.dataset_home=os.path.join(datasets_root, self.dataset_name, 'raw')\n",
    "        \n",
    "        print('Dataset path: ', self.dataset_home)\n",
    "        \n",
    "    def init_part(self):\n",
    "        self.part = {\n",
    "            'X'        : None,\n",
    "            'Y'        : None,\n",
    "            'X_train'  : None,\n",
    "            'Y_train'  : None,\n",
    "            'X_valid'  : None,\n",
    "            'Y_valid'  : None,\n",
    "            'X_test'   : None,\n",
    "            'Y_test'   : None,\n",
    "            'train'    : None,\n",
    "            'test'     : None,\n",
    "            'valid'    : None,\n",
    "        }\n",
    "        \n",
    "        self.index={\n",
    "            'train'    : 0,\n",
    "            'test'     : 0,\n",
    "            'valid'    : 0,\n",
    "        }"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
