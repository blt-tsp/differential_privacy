{
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/blt-tsp/differential_privacy/blob/main/diff_privacy_CNN.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "X72Y9iiS8pLH",
        "outputId": "69ad188c-92a1-4b3e-a792-1e4cc9bbf003"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Requirement already satisfied: opacus in /usr/local/lib/python3.10/dist-packages (1.4.0)\n",
            "Requirement already satisfied: numpy>=1.15 in /usr/local/lib/python3.10/dist-packages (from opacus) (1.23.5)\n",
            "Requirement already satisfied: torch>=1.13 in /usr/local/lib/python3.10/dist-packages (from opacus) (2.1.0+cu121)\n",
            "Requirement already satisfied: scipy>=1.2 in /usr/local/lib/python3.10/dist-packages (from opacus) (1.11.4)\n",
            "Requirement already satisfied: opt-einsum>=3.3.0 in /usr/local/lib/python3.10/dist-packages (from opacus) (3.3.0)\n",
            "Requirement already satisfied: filelock in /usr/local/lib/python3.10/dist-packages (from torch>=1.13->opacus) (3.13.1)\n",
            "Requirement already satisfied: typing-extensions in /usr/local/lib/python3.10/dist-packages (from torch>=1.13->opacus) (4.5.0)\n",
            "Requirement already satisfied: sympy in /usr/local/lib/python3.10/dist-packages (from torch>=1.13->opacus) (1.12)\n",
            "Requirement already satisfied: networkx in /usr/local/lib/python3.10/dist-packages (from torch>=1.13->opacus) (3.2.1)\n",
            "Requirement already satisfied: jinja2 in /usr/local/lib/python3.10/dist-packages (from torch>=1.13->opacus) (3.1.3)\n",
            "Requirement already satisfied: fsspec in /usr/local/lib/python3.10/dist-packages (from torch>=1.13->opacus) (2023.6.0)\n",
            "Requirement already satisfied: triton==2.1.0 in /usr/local/lib/python3.10/dist-packages (from torch>=1.13->opacus) (2.1.0)\n",
            "Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.10/dist-packages (from jinja2->torch>=1.13->opacus) (2.1.3)\n",
            "Requirement already satisfied: mpmath>=0.19 in /usr/local/lib/python3.10/dist-packages (from sympy->torch>=1.13->opacus) (1.3.0)\n"
          ]
        }
      ],
      "source": [
        "!pip install opacus"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "background_save": true,
          "base_uri": "https://localhost:8080/"
        },
        "id": "O_z8JEmWN5ZV",
        "outputId": "d2c0baec-09cf-4ab0-9b7d-985d90343cc8"
      },
      "outputs": [
        {
          "name": "stderr",
          "output_type": "stream",
          "text": [
            "/usr/local/lib/python3.10/dist-packages/opacus/privacy_engine.py:142: UserWarning: Secure RNG turned off. This is perfectly fine for experimentation as it allows for much faster training performance, but remember to turn it on and retrain one last time before production with ``secure_mode`` turned on.\n",
            "  warnings.warn(\n",
            "/usr/local/lib/python3.10/dist-packages/opacus/accountants/analysis/rdp.py:332: UserWarning: Optimal order is the largest alpha. Please consider expanding the range of alphas to get a tighter privacy bound.\n",
            "  warnings.warn(\n",
            "/usr/local/lib/python3.10/dist-packages/opacus/accountants/analysis/prv/prvs.py:50: RuntimeWarning: invalid value encountered in log\n",
            "  z = np.log((np.exp(t) + q - 1) / q)\n",
            "  0%|          | 0/20 [00:00<?, ?it/s]/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py:1359: UserWarning: Using a non-full backward hook when the forward contains multiple autograd Nodes is deprecated and will be removed in future versions. This hook will be missing some grad_input. Please use register_full_backward_hook to get the documented behavior.\n",
            "  warnings.warn(\"Using a non-full backward hook when the forward contains multiple autograd Nodes \"\n",
            "100%|██████████| 20/20 [01:15<00:00,  3.79s/it]\n",
            "/usr/local/lib/python3.10/dist-packages/opacus/privacy_engine.py:142: UserWarning: Secure RNG turned off. This is perfectly fine for experimentation as it allows for much faster training performance, but remember to turn it on and retrain one last time before production with ``secure_mode`` turned on.\n",
            "  warnings.warn(\n",
            "/usr/local/lib/python3.10/dist-packages/opacus/accountants/analysis/rdp.py:332: UserWarning: Optimal order is the largest alpha. Please consider expanding the range of alphas to get a tighter privacy bound.\n",
            "  warnings.warn(\n"
          ]
        },
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "(ε = 0.01, δ = 1e-05)\n"
          ]
        },
        {
          "name": "stderr",
          "output_type": "stream",
          "text": [
            "/usr/local/lib/python3.10/dist-packages/opacus/accountants/analysis/prv/prvs.py:50: RuntimeWarning: invalid value encountered in log\n",
            "  z = np.log((np.exp(t) + q - 1) / q)\n",
            "  0%|          | 0/20 [00:00<?, ?it/s]/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py:1359: UserWarning: Using a non-full backward hook when the forward contains multiple autograd Nodes is deprecated and will be removed in future versions. This hook will be missing some grad_input. Please use register_full_backward_hook to get the documented behavior.\n",
            "  warnings.warn(\"Using a non-full backward hook when the forward contains multiple autograd Nodes \"\n",
            "100%|██████████| 20/20 [01:14<00:00,  3.73s/it]\n",
            "/usr/local/lib/python3.10/dist-packages/opacus/privacy_engine.py:142: UserWarning: Secure RNG turned off. This is perfectly fine for experimentation as it allows for much faster training performance, but remember to turn it on and retrain one last time before production with ``secure_mode`` turned on.\n",
            "  warnings.warn(\n",
            "/usr/local/lib/python3.10/dist-packages/opacus/accountants/analysis/rdp.py:332: UserWarning: Optimal order is the largest alpha. Please consider expanding the range of alphas to get a tighter privacy bound.\n",
            "  warnings.warn(\n",
            "/usr/local/lib/python3.10/dist-packages/opacus/accountants/analysis/prv/prvs.py:50: RuntimeWarning: invalid value encountered in log\n",
            "  z = np.log((np.exp(t) + q - 1) / q)\n"
          ]
        },
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "(ε = 0.05, δ = 1e-05)\n"
          ]
        },
        {
          "name": "stderr",
          "output_type": "stream",
          "text": [
            "  0%|          | 0/20 [00:00<?, ?it/s]/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py:1359: UserWarning: Using a non-full backward hook when the forward contains multiple autograd Nodes is deprecated and will be removed in future versions. This hook will be missing some grad_input. Please use register_full_backward_hook to get the documented behavior.\n",
            "  warnings.warn(\"Using a non-full backward hook when the forward contains multiple autograd Nodes \"\n",
            "100%|██████████| 20/20 [01:13<00:00,  3.69s/it]\n",
            "/usr/local/lib/python3.10/dist-packages/opacus/privacy_engine.py:142: UserWarning: Secure RNG turned off. This is perfectly fine for experimentation as it allows for much faster training performance, but remember to turn it on and retrain one last time before production with ``secure_mode`` turned on.\n",
            "  warnings.warn(\n"
          ]
        },
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "(ε = 0.10, δ = 1e-05)\n"
          ]
        },
        {
          "name": "stderr",
          "output_type": "stream",
          "text": [
            "/usr/local/lib/python3.10/dist-packages/opacus/accountants/analysis/rdp.py:332: UserWarning: Optimal order is the largest alpha. Please consider expanding the range of alphas to get a tighter privacy bound.\n",
            "  warnings.warn(\n",
            "/usr/local/lib/python3.10/dist-packages/opacus/accountants/analysis/prv/prvs.py:50: RuntimeWarning: invalid value encountered in log\n",
            "  z = np.log((np.exp(t) + q - 1) / q)\n",
            "  0%|          | 0/20 [00:00<?, ?it/s]/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py:1359: UserWarning: Using a non-full backward hook when the forward contains multiple autograd Nodes is deprecated and will be removed in future versions. This hook will be missing some grad_input. Please use register_full_backward_hook to get the documented behavior.\n",
            "  warnings.warn(\"Using a non-full backward hook when the forward contains multiple autograd Nodes \"\n",
            "100%|██████████| 20/20 [01:12<00:00,  3.60s/it]\n",
            "/usr/local/lib/python3.10/dist-packages/opacus/privacy_engine.py:142: UserWarning: Secure RNG turned off. This is perfectly fine for experimentation as it allows for much faster training performance, but remember to turn it on and retrain one last time before production with ``secure_mode`` turned on.\n",
            "  warnings.warn(\n",
            "/usr/local/lib/python3.10/dist-packages/opacus/accountants/analysis/rdp.py:332: UserWarning: Optimal order is the largest alpha. Please consider expanding the range of alphas to get a tighter privacy bound.\n",
            "  warnings.warn(\n",
            "/usr/local/lib/python3.10/dist-packages/opacus/accountants/analysis/prv/prvs.py:50: RuntimeWarning: invalid value encountered in log\n",
            "  z = np.log((np.exp(t) + q - 1) / q)\n"
          ]
        },
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "(ε = 1.00, δ = 1e-05)\n"
          ]
        },
        {
          "name": "stderr",
          "output_type": "stream",
          "text": [
            "  0%|          | 0/20 [00:00<?, ?it/s]/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py:1359: UserWarning: Using a non-full backward hook when the forward contains multiple autograd Nodes is deprecated and will be removed in future versions. This hook will be missing some grad_input. Please use register_full_backward_hook to get the documented behavior.\n",
            "  warnings.warn(\"Using a non-full backward hook when the forward contains multiple autograd Nodes \"\n",
            "100%|██████████| 20/20 [01:09<00:00,  3.50s/it]\n"
          ]
        },
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "(ε = 10.00, δ = 1e-05)\n"
          ]
        },
        {
          "data": {
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAAkAAAAHHCAYAAABXx+fLAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/bCgiHAAAACXBIWXMAAA9hAAAPYQGoP6dpAABM40lEQVR4nO3de1xUdf7H8fdw9wZeUC6K4C0vmXjFS1q50bplmpWpPVwxNa1Ns2Jz0zY17UJZ689S03LT7JcpW6nr1uZmZJm/TEsitVKTVLwBkgmIhTqc3x+zjIyAzuDMHGRez8fjPGC+c86Z70zmvP18v99zLIZhGAIAAPAhfmZ3AAAAwNsIQAAAwOcQgAAAgM8hAAEAAJ9DAAIAAD6HAAQAAHwOAQgAAPgcAhAAAPA5BCAAAOBzCEAAUMM8+eSTslgsDm1xcXG65557zOkQUA0RgIArzCuvvCKLxaKePXua3RVcwoEDB2SxWCrdnnvuObO7CPisALM7AMA1K1asUFxcnLZt26Z9+/apdevWZncJl3D33XfrlltuKdfepUsXj7zeE088oalTp3rk3EBNQQACriD79+/XF198odWrV+u+++7TihUrNHPmTLO7VaGioiLVqVPH7G5UC127dtUf//hHr71eQECAAgL46x24GIbAgCvIihUr1KBBAw0cOFBDhw7VihUrKtzv5MmTeuSRRxQXF6fg4GA1a9ZMSUlJysvLs+/z22+/6cknn9RVV12lkJAQRUVF6Y477lBmZqYk6dNPP5XFYtGnn37qcO7SYZ033njD3nbPPfeobt26yszM1C233KJ69epp5MiRkqTPP/9cd911l5o3b67g4GDFxMTokUce0a+//lqu37t379awYcPUuHFj1apVS23bttVf//pXSdLGjRtlsVi0Zs2acse9/fbbslgs2rJlS4Wfx9dffy2LxaLly5eXe+4///mPLBaL3n//fUlSYWGhHn74Yftn16RJE910001KT0+v8NzuEhcXp1tvvVUfffSROnfurJCQEHXo0EGrV6922O/s2bOaNWuW2rRpo5CQEDVq1Eh9+/bVhg0b7PtUNAeoIj/99JPuuusuNWzYULVr11avXr30wQcfOOxT+ufgH//4h5555hk1a9ZMISEhuvHGG7Vv3z73vHnABPwTAbiCrFixQnfccYeCgoJ09913a9GiRfrqq6/Uo0cP+z6nTp1Sv3799MMPP2js2LHq2rWr8vLytG7dOh0+fFjh4eGyWq269dZblZaWphEjRuihhx5SYWGhNmzYoF27dqlVq1Yu9+3cuXMaMGCA+vbtqxdffFG1a9eWJL3zzjs6ffq0/vSnP6lRo0batm2b5s+fr8OHD+udd96xH79jxw7169dPgYGBmjBhguLi4pSZmal//etfeuaZZ3TDDTcoJiZGK1as0O23317uc2nVqpV69+5dYd+6d++uli1b6h//+IdGjx7t8FxqaqoaNGigAQMGSJLuv/9+vfvuu5o0aZI6dOign3/+WZs3b9YPP/ygrl27uvy5SNLp06cdwmep+vXrO1RqfvzxRw0fPlz333+/Ro8erWXLlumuu+7S+vXrddNNN0myhZuUlBTde++9SkhIUEFBgb7++mulp6fb93FGTk6O+vTpo9OnT2vy5Mlq1KiRli9frsGDB+vdd98t9xk/99xz8vPz06OPPqr8/HzNmTNHI0eO1NatW6v0mQCmMwBcEb7++mtDkrFhwwbDMAyjpKTEaNasmfHQQw857DdjxgxDkrF69epy5ygpKTEMwzCWLl1qSDLmzp1b6T4bN240JBkbN250eH7//v2GJGPZsmX2ttGjRxuSjKlTp5Y73+nTp8u1paSkGBaLxTh48KC97brrrjPq1avn0Fa2P4ZhGNOmTTOCg4ONkydP2ttyc3ONgIAAY+bMmeVep6xp06YZgYGBxokTJ+xtxcXFRv369Y2xY8fa28LCwoyJEyde9FzOKv2sKtu2bNli3zc2NtaQZLz33nv2tvz8fCMqKsro0qWLvS0+Pt4YOHDgRV935syZxoV/vcfGxhqjR4+2P3744YcNScbnn39ubyssLDRatGhhxMXFGVar1TCM838O2rdvbxQXF9v3femllwxJxs6dO137UIBqgiEw4AqxYsUKRUREqH///pIki8Wi4cOHa9WqVbJarfb93nvvPcXHx5f7F3zpMaX7hIeH68EHH6x0n6r405/+VK6tVq1a9t+LioqUl5enPn36yDAMffPNN5Kk48ePa9OmTRo7dqyaN29eaX+SkpJUXFysd999196Wmpqqc+fOXXKOzfDhw3X27FmHIaWPPvpIJ0+e1PDhw+1t9evX19atW3X06FEn3/WlTZgwQRs2bCi3dejQwWG/6Ohoh/9uoaGhSkpK0jfffKPs7Gx7/7777jv9+OOPl9Wnf//730pISFDfvn3tbXXr1tWECRN04MABff/99w77jxkzRkFBQfbH/fr1k2QbRgOuRAQg4ApgtVq1atUq9e/fX/v379e+ffu0b98+9ezZUzk5OUpLS7Pvm5mZqY4dO170fJmZmWrbtq1bJ8oGBASoWbNm5dqzsrJ0zz33qGHDhqpbt64aN26s66+/XpKUn58v6fyX6KX63a5dO/Xo0cNh7tOKFSvUq1evS66Gi4+PV7t27ZSammpvS01NVXh4uH73u9/Z2+bMmaNdu3YpJiZGCQkJevLJJy/7S75NmzZKTEwst4WGhjrs17p163IB9KqrrpJkm3slSbNnz9bJkyd11VVX6ZprrtGUKVO0Y8cOl/t08OBBtW3btlx7+/bt7c+XdWEwbdCggSTpl19+cfm1geqAAARcAT755BMdO3ZMq1atUps2bezbsGHDJKnSydCXo7JKUNlqU1nBwcHy8/Mrt+9NN92kDz74QI899pjWrl2rDRs22CdQl5SUuNyvpKQkffbZZzp8+LAyMzP15ZdfOr3Cavjw4dq4caPy8vJUXFysdevW6c4773QIgsOGDdNPP/2k+fPnKzo6Wi+88IKuvvpqffjhhy731ROuu+46ZWZmaunSperYsaP+/ve/q2vXrvr73//u0df19/evsN0wDI++LuApTIIGrgArVqxQkyZNtHDhwnLPrV69WmvWrNHixYtVq1YttWrVSrt27bro+Vq1aqWtW7fq7NmzCgwMrHCf0n/hnzx50qH9wsrAxezcuVN79+7V8uXLlZSUZG8vu2JJklq2bClJl+y3JI0YMULJyclauXKlfv31VwUGBjoMYV3M8OHDNWvWLL333nuKiIhQQUGBRowYUW6/qKgoPfDAA3rggQeUm5urrl276plnntHNN9/s1OtU1b59+2QYhkP43Lt3ryTbKrFSDRs21JgxYzRmzBidOnVK1113nZ588knde++9Tr9WbGys9uzZU6599+7d9ueBmowKEFDN/frrr1q9erVuvfVWDR06tNw2adIkFRYWat26dZKkO++8U99++22Fy8VL/7V+5513Ki8vTwsWLKh0n9jYWPn7+2vTpk0Oz7/yyitO9720alC2SmAYhl566SWH/Ro3bqzrrrtOS5cuVVZWVoX9KRUeHq6bb75Zb731llasWKE//OEPCg8Pd6o/7du31zXXXKPU1FSlpqYqKipK1113nf15q9VqH5Yr1aRJE0VHR6u4uNjelpeXp927d+v06dNOva6zjh496vDfraCgQG+++aY6d+6syMhISdLPP//scEzdunXVunVrh/4545ZbbtG2bdscLh1QVFSk1157TXFxceXmJwE1DRUgoJpbt26dCgsLNXjw4Aqf79Wrlxo3bqwVK1Zo+PDhmjJlit59913dddddGjt2rLp166YTJ05o3bp1Wrx4seLj45WUlKQ333xTycnJ2rZtm/r166eioiJ9/PHHeuCBB3TbbbcpLCxMd911l+bPny+LxaJWrVrp/fffV25urtN9b9eunVq1aqVHH31UR44cUWhoqN57770K5428/PLL6tu3r7p27aoJEyaoRYsWOnDggD744ANlZGQ47JuUlKShQ4dKkp566innP0zZqkAzZsxQSEiIxo0b5zBsV1hYqGbNmmno0KGKj49X3bp19fHHH+urr77S3/72N/t+CxYs0KxZs7Rx40bdcMMNl3zN9PR0vfXWW+XaL1y6f9VVV2ncuHH66quvFBERoaVLlyonJ0fLli2z79OhQwfdcMMN6tatmxo2bKivv/7avmzfFVOnTtXKlSt18803a/LkyWrYsKGWL1+u/fv367333is3nAnUOOYtQAPgjEGDBhkhISFGUVFRpfvcc889RmBgoJGXl2cYhmH8/PPPxqRJk4ymTZsaQUFBRrNmzYzRo0fbnzcM2/L0v/71r0aLFi2MwMBAIzIy0hg6dKiRmZlp3+f48ePGnXfeadSuXdto0KCBcd999xm7du2qcBl8nTp1Kuzb999/byQmJhp169Y1wsPDjfHjxxvffvttuXMYhmHs2rXLuP3224369esbISEhRtu2bY3p06eXO2dxcbHRoEEDIywszPj111+d+RjtfvzxR/sy9M2bN5c775QpU4z4+HijXr16Rp06dYz4+HjjlVdecdivdJn5hZcIuNCllsGXXZYeGxtrDBw40PjPf/5jdOrUyQgODjbatWtnvPPOOw7nfPrpp42EhASjfv36Rq1atYx27doZzzzzjHHmzJly/SvrwmXwhmEYmZmZxtChQ+2fd0JCgvH+++877FO6DP7CflR0OQTgSmIxDGawAbiynDt3TtHR0Ro0aJBef/11s7vjFnFxcerYsaP9itQAPIsaJ4Arztq1a3X8+HGHidUA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\n",
            "text/plain": [
              "<Figure size 640x480 with 1 Axes>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        }
      ],
      "source": [
        "import torch\n",
        "import torch.nn as nn\n",
        "import torch.optim as optim\n",
        "import torch.nn.functional as F\n",
        "\n",
        "import pandas as pd\n",
        "from torch.utils.data import DataLoader, TensorDataset, random_split\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.preprocessing import StandardScaler\n",
        "from sklearn.metrics import accuracy_score\n",
        "\n",
        "from tqdm import tqdm\n",
        "import matplotlib.pyplot as plt\n",
        "\n",
        "from opacus import PrivacyEngine\n",
        "from opacus.layers import DPLSTM\n",
        "\n",
        "# Step 1: Define a Simpler CNN Model\n",
        "class TimeSeriesClassification(nn.Module):\n",
        "    def __init__(self, hidden_size, num_layers):\n",
        "        super(TimeSeriesClassification, self).__init__()\n",
        "\n",
        "        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, padding=1, stride=1)\n",
        "        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1, stride=1)\n",
        "        self.conv3 = nn.Conv1d(128, 64, kernel_size=3, padding=1, stride=1)\n",
        "\n",
        "        self.pool = nn.MaxPool1d(kernel_size=1)\n",
        "\n",
        "        # Capturing temporal dependencies\n",
        "        self.rnn = DPLSTM(input_size=64, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)\n",
        "\n",
        "        self.fc1 = nn.Linear(hidden_size, 64)\n",
        "        self.fc2 = nn.Linear(64, 2)\n",
        "\n",
        "        self.dropout = nn.Dropout(0.1)\n",
        "\n",
        "    def forward(self, x):\n",
        "        x = self.pool(F.relu(self.conv1(x)))\n",
        "        x = self.pool(F.relu(self.conv2(x)))\n",
        "        x = self.pool(F.relu(self.conv3(x)))\n",
        "\n",
        "        # Rearranging for LSTM\n",
        "        x = x.permute(0, 2, 1)\n",
        "        _, (h_n, _) = self.rnn(x)\n",
        "\n",
        "        x = F.relu(self.fc1(h_n[-1, :, :]))\n",
        "        x = self.dropout(x)\n",
        "        x = self.fc2(x)\n",
        "\n",
        "        return x\n",
        "\n",
        "# Step 2: Load and Preprocess the Dataset\n",
        "df = pd.read_csv('Fuel_data.csv')\n",
        "\n",
        "features = df[['IPG2211A2N']].values.astype(float)\n",
        "labels = df['FLAG'].values.astype(int)\n",
        "\n",
        "scaler = StandardScaler()\n",
        "features = scaler.fit_transform(features)\n",
        "\n",
        "# Convert to PyTorch tensors\n",
        "features_tensor = torch.FloatTensor(features).unsqueeze(1)  # Add a channel dimension\n",
        "labels_tensor = torch.LongTensor(labels)\n",
        "\n",
        "# Step 3: Split the Data into Training, Validation, and Test Sets\n",
        "dataset_size = len(features_tensor)\n",
        "train_size = int(0.6 * dataset_size)\n",
        "val_size = int(0.2 * dataset_size)\n",
        "test_size = dataset_size - train_size - val_size\n",
        "\n",
        "train_dataset, val_dataset, test_dataset = random_split(\n",
        "    TensorDataset(features_tensor, labels_tensor),\n",
        "    [train_size, val_size, test_size]\n",
        ")\n",
        "\n",
        "# Create DataLoader for Training, Validation, and Testing\n",
        "train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)\n",
        "val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)\n",
        "test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)\n",
        "\n",
        "# Differential Privacy - Stochastic Gradient Descent\n",
        "\n",
        "def loss_fn(logits, labels):\n",
        "    return F.cross_entropy(logits, labels)\n",
        "\n",
        "def dp_train(model, epochs, train_loader, max_grad_norm, epsilon, delta):\n",
        "\n",
        "    #DP training on train set\n",
        "    privacy_engine = PrivacyEngine()\n",
        "\n",
        "    privacy_engine.make_private_with_epsilon(\n",
        "        epochs=epochs,\n",
        "        module = model,\n",
        "        optimizer = optimizer,\n",
        "        data_loader = train_loader,\n",
        "        max_grad_norm=max_grad_norm,\n",
        "        target_epsilon=epsilon,\n",
        "        target_delta=delta\n",
        "    )\n",
        "\n",
        "    for epoch in tqdm(range(epochs)):\n",
        "        model.train()\n",
        "\n",
        "        for batch_x, batch_y in train_loader:\n",
        "            optimizer.zero_grad()\n",
        "            batch_x.to(device), batch_y.to(device)\n",
        "\n",
        "            # Forward pass\n",
        "            logits = model(batch_x.float())\n",
        "            loss = loss_fn(logits, batch_y)\n",
        "\n",
        "            # Backward pass\n",
        "            loss.backward()\n",
        "\n",
        "            # Optimize with DP-SGD\n",
        "            optimizer.step()\n",
        "\n",
        "def dp_eval(model, test_loader):\n",
        "    # Get the number of input channels from the first batch in val_loader\n",
        "    num_input_channels = next(iter(test_loader))[0].shape[1]\n",
        "\n",
        "    # Evaluation on test set\n",
        "    model.eval()\n",
        "    val_preds, val_labels = [], []\n",
        "\n",
        "    with torch.no_grad():\n",
        "        for batch_x, batch_y in test_loader:\n",
        "            batch_x.to(device), batch_y.to(device)\n",
        "            logits = model(batch_x.float())\n",
        "            val_preds.extend(torch.argmax(logits, dim=1).tolist())\n",
        "            val_labels.extend(batch_y.tolist())\n",
        "\n",
        "    val_accuracy = accuracy_score(val_labels, val_preds)\n",
        "    return val_accuracy\n",
        "\n",
        "# Step 6: Run the model\n",
        "if __name__ == \"__main__\":\n",
        "    epochs = 20\n",
        "\n",
        "    # DP hyperparameters\n",
        "    max_grad_norm = 1.0\n",
        "    delta_value = 1e-5\n",
        "\n",
        "    # RNN parameters\n",
        "    hidden_size=64\n",
        "    num_layers=2\n",
        "\n",
        "    num_input_channels=1\n",
        "    device = \"cuda\" if torch.cuda.is_available() else \"cpu\"\n",
        "\n",
        "    # Plotting the accuracy with change of epsilon\n",
        "    epsilon_values = [0.01, 0.05, 0.1, 1.0, 10]  # Adjust as needed\n",
        "    accuracies = []\n",
        "\n",
        "    for epsilon in epsilon_values:\n",
        "        model = TimeSeriesClassification(hidden_size=hidden_size, num_layers=num_layers).to(device)\n",
        "        optimizer = optim.Adam(model.parameters(), lr=1e-4)\n",
        "\n",
        "        # Train and eval\n",
        "        dp_train(model, epochs=epochs, train_loader=train_loader, max_grad_norm=max_grad_norm, epsilon=epsilon, delta=delta_value)\n",
        "        accuracy = dp_eval(model, test_loader=test_loader)\n",
        "        accuracies.append(accuracy)\n",
        "\n",
        "        print(f\"(ε = {epsilon:.2f}, δ = {delta_value})\")\n",
        "\n",
        "    # Plotting the accuracy vs. epsilon\n",
        "    plt.plot(epsilon_values, accuracies, marker='+', color='r')\n",
        "    plt.xlabel('Epsilon (ε)')\n",
        "    plt.ylabel('Accuracy')\n",
        "    plt.title('Accuracy vs. Epsilon')\n",
        "    plt.show()"
      ]
    }
  ],
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyNYQSVS55gFoRHBSe+UPbzj",
      "include_colab_link": true
    },
    "kernelspec": {
      "display_name": "Python 3",
      "name": "python3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 0
}