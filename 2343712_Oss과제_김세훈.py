{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyM+LuNJDITnNdgWSDaXwysa",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/saehoon3/Oss-assignment/blob/feature-wine-task/2343712_Oss%EA%B3%BC%EC%A0%9C_%EA%B9%80%EC%84%B8%ED%9B%88.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 1,
      "metadata": {
        "id": "vkxn72vqWfSX"
      },
      "outputs": [],
      "source": [
        "import pandas as pd\n",
        "from sklearn.datasets import load_wine\n",
        "from sklearn.model_selection import train_test_split"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "#데이터셋 로드\n",
        "wine = load_wine()\n",
        "print(wine)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "sBeWxI8ugbzi",
        "outputId": "e513d25b-9553-41de-e957-2d1b18a9670b"
      },
      "execution_count": 5,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "{'data': array([[1.423e+01, 1.710e+00, 2.430e+00, ..., 1.040e+00, 3.920e+00,\n",
            "        1.065e+03],\n",
            "       [1.320e+01, 1.780e+00, 2.140e+00, ..., 1.050e+00, 3.400e+00,\n",
            "        1.050e+03],\n",
            "       [1.316e+01, 2.360e+00, 2.670e+00, ..., 1.030e+00, 3.170e+00,\n",
            "        1.185e+03],\n",
            "       ...,\n",
            "       [1.327e+01, 4.280e+00, 2.260e+00, ..., 5.900e-01, 1.560e+00,\n",
            "        8.350e+02],\n",
            "       [1.317e+01, 2.590e+00, 2.370e+00, ..., 6.000e-01, 1.620e+00,\n",
            "        8.400e+02],\n",
            "       [1.413e+01, 4.100e+00, 2.740e+00, ..., 6.100e-01, 1.600e+00,\n",
            "        5.600e+02]]), 'target': array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,\n",
            "       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,\n",
            "       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1,\n",
            "       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,\n",
            "       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,\n",
            "       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2,\n",
            "       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,\n",
            "       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,\n",
            "       2, 2]), 'frame': None, 'target_names': array(['class_0', 'class_1', 'class_2'], dtype='<U7'), 'DESCR': '.. _wine_dataset:\\n\\nWine recognition dataset\\n------------------------\\n\\n**Data Set Characteristics:**\\n\\n:Number of Instances: 178\\n:Number of Attributes: 13 numeric, predictive attributes and the class\\n:Attribute Information:\\n    - Alcohol\\n    - Malic acid\\n    - Ash\\n    - Alcalinity of ash\\n    - Magnesium\\n    - Total phenols\\n    - Flavanoids\\n    - Nonflavanoid phenols\\n    - Proanthocyanins\\n    - Color intensity\\n    - Hue\\n    - OD280/OD315 of diluted wines\\n    - Proline\\n    - class:\\n        - class_0\\n        - class_1\\n        - class_2\\n\\n:Summary Statistics:\\n\\n============================= ==== ===== ======= =====\\n                                Min   Max   Mean     SD\\n============================= ==== ===== ======= =====\\nAlcohol:                      11.0  14.8    13.0   0.8\\nMalic Acid:                   0.74  5.80    2.34  1.12\\nAsh:                          1.36  3.23    2.36  0.27\\nAlcalinity of Ash:            10.6  30.0    19.5   3.3\\nMagnesium:                    70.0 162.0    99.7  14.3\\nTotal Phenols:                0.98  3.88    2.29  0.63\\nFlavanoids:                   0.34  5.08    2.03  1.00\\nNonflavanoid Phenols:         0.13  0.66    0.36  0.12\\nProanthocyanins:              0.41  3.58    1.59  0.57\\nColour Intensity:              1.3  13.0     5.1   2.3\\nHue:                          0.48  1.71    0.96  0.23\\nOD280/OD315 of diluted wines: 1.27  4.00    2.61  0.71\\nProline:                       278  1680     746   315\\n============================= ==== ===== ======= =====\\n\\n:Missing Attribute Values: None\\n:Class Distribution: class_0 (59), class_1 (71), class_2 (48)\\n:Creator: R.A. Fisher\\n:Donor: Michael Marshall (MARSHALL%PLU@io.arc.nasa.gov)\\n:Date: July, 1988\\n\\nThis is a copy of UCI ML Wine recognition datasets.\\nhttps://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data\\n\\nThe data is the results of a chemical analysis of wines grown in the same\\nregion in Italy by three different cultivators. There are thirteen different\\nmeasurements taken for different constituents found in the three types of\\nwine.\\n\\nOriginal Owners:\\n\\nForina, M. et al, PARVUS -\\nAn Extendible Package for Data Exploration, Classification and Correlation.\\nInstitute of Pharmaceutical and Food Analysis and Technologies,\\nVia Brigata Salerno, 16147 Genoa, Italy.\\n\\nCitation:\\n\\nLichman, M. (2013). UCI Machine Learning Repository\\n[https://archive.ics.uci.edu/ml]. Irvine, CA: University of California,\\nSchool of Information and Computer Science.\\n\\n.. dropdown:: References\\n\\n    (1) S. Aeberhard, D. Coomans and O. de Vel,\\n    Comparison of Classifiers in High Dimensional Settings,\\n    Tech. Rep. no. 92-02, (1992), Dept. of Computer Science and Dept. of\\n    Mathematics and Statistics, James Cook University of North Queensland.\\n    (Also submitted to Technometrics).\\n\\n    The data was used with many others for comparing various\\n    classifiers. The classes are separable, though only RDA\\n    has achieved 100% correct classification.\\n    (RDA : 100%, QDA 99.4%, LDA 98.9%, 1NN 96.1% (z-transformed data))\\n    (All results using the leave-one-out technique)\\n\\n    (2) S. Aeberhard, D. Coomans and O. de Vel,\\n    \"THE CLASSIFICATION PERFORMANCE OF RDA\"\\n    Tech. Rep. no. 92-01, (1992), Dept. of Computer Science and Dept. of\\n    Mathematics and Statistics, James Cook University of North Queensland.\\n    (Also submitted to Journal of Chemometrics).\\n', 'feature_names': ['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium', 'total_phenols', 'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins', 'color_intensity', 'hue', 'od280/od315_of_diluted_wines', 'proline']}\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# 훈련 데이터와 테스트 데이터 분리\n",
        "X = wine.data\n",
        "y = wine.target\n",
        "\n",
        "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
        "\n",
        "print(X_train.shape)\n",
        "print(X_test.shape)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "nvVh101VgmUY",
        "outputId": "5ddb79d2-1464-4587-9dd0-4faec19914d7"
      },
      "execution_count": 7,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "(142, 13)\n",
            "(36, 13)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.ensemble import RandomForestClassifier\n",
        "from sklearn.metrics import accuracy_score, classification_report\n",
        "\n",
        "model = RandomForestClassifier(random_state=42)\n",
        "model.fit(X_train, y_train)\n",
        "\n",
        "y_pred = model.predict(X_test)\n",
        "accuracy = accuracy_score(y_test, y_pred)\n",
        "print(accuracy)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "0mS877HtspCF",
        "outputId": "073fb033-0bdd-4688-ed39-33d773e0d2d3"
      },
      "execution_count": 10,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "1.0\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "new_model = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)\n",
        "new_model.fit(X_train, y_train)\n",
        "\n",
        "y_pred = new_model.fit(X_train, y_train).predict(X_test)\n",
        "accuracy = accuracy_score(y_test, y_pred)\n",
        "print(accuracy)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "x2bkiJ9EvrTz",
        "outputId": "071a9d2d-d2cb-4b9a-82d0-1f3bd2d97492"
      },
      "execution_count": 13,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "0.9444444444444444\n"
          ]
        }
      ]
    }
  ]
}