from utils import get_af2_emb

import click
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score
import joblib
import mord

def train_monomer(
    training_data, af2_embeddings_dir: str = '../calc', C=10, balanced=0, dual=1,
    ensemble_size=1, use_pairwise=True, use_scaler=True, output_dir: str = None
) -> tuple:
    """
    Train an ensemble of logistic regression models. Predicts whether a protein is a monomer or a homomer 

    Parameters:
        training_data (str): Path to training data.
        c (int, optional): Regularization parameter. Defaults to 10.
        balanced (int, optional): Whether to balance class weights. Defaults to 0.
        dual (int, optional): Whether to use dual formulation. Defaults to 1.
        ensemble_size (int, optional): Number of ensemble models. Defaults to 1.
        use_pairwise (bool, optional): Whether to use pairwise embeddings. Defaults to True.
        use_scaler (bool, optional): Whether to use data scaling. Defaults to True.
        output_dir (str, optional): Directory to save the output files. Defaults to None.

    Returns:
        tuple: Dictionary containing training results, trained models, and DataFrame with results.
    """

    # Load the training data
    df = pd.read_csv(training_data, sep="\t")
    # Create a binary state column based on the number of chains
    df['state'] = [1 if i > 1 else 0 for i in df['chains']]
    # Encode the state labels
    le = LabelEncoder()
    df['y'] = le.fit_transform(df['state'])
    print(df)

    # Initialize arrays to store results and internal representations
    results = np.zeros((ensemble_size, 5, len(df), 2))  # Adjusted to 2 classes
    internal_representations = np.zeros((ensemble_size, 5, len(df)))  # Adjusted to 1 value per sample for binary classification
    model = {}
    total_iterations = ensemble_size * 5
    iteration = 0

    # Loop over the ensemble size
    for j in range(ensemble_size):
        # Loop over the 5 AF2 models
        for i in range(5):
            iteration += 1
            print(f"Training progress: {iteration}/{total_iterations} iterations completed.")
            # Get AF2 embeddings for each PDB ID
            X = np.asarray([get_af2_emb(af2_embeddings_dir, id_=id_, model_id=i, use_pairwise=use_pairwise) for id_ in df['pdb']])
            y = df['y'].values
            # Initialize KFold cross-validation
            cv = KFold(n_splits=5, shuffle=True)

            # Loop over the cross-validation splits
            for k, (tr_idx, te_idx) in enumerate(cv.split(X, y)):
                X_tr, X_te = X[tr_idx], X[te_idx]
                y_tr, y_te = y[tr_idx], y[te_idx]

                # Scale the data if use_scaler is True
                if use_scaler:
                    sc = StandardScaler()
                    X_tr = sc.fit_transform(X_tr)
                    X_te = sc.transform(X_te)
                    model[f"scaler_{j}_{i}_{k}"] = sc

                # Initialize and train the logistic regression model
                clf = LogisticRegression(C=C, max_iter=1000, solver='liblinear',
                                         dual=False if dual == 0 else True,
                                         class_weight='balanced' if balanced == 1 else None)
                clf.fit(X_tr, y_tr)
                # Store the predicted probabilities and internal representations
                results[j, i, te_idx, :] = clf.predict_proba(X_te)
                internal_representations[j, i, te_idx] = clf.decision_function(X_te)
                model[f"clf_{j}_{i}_{k}"] = clf

    # Calculate the final predictions and evaluation metrics
    y_pred_bin = results.mean(axis=0).mean(axis=0).argmax(axis=1)
    results_ = {}
    results_["accuracy"] = accuracy_score(y, y_pred_bin)
    results_["f1"] = f1_score(y, y_pred_bin, average='macro')
    df["y_pred"] = y_pred_bin
    df["prob_monomer"] = results.mean(axis=0).mean(axis=0)[:, 0]
    df["prob_homomer"] = results.mean(axis=0).mean(axis=0)[:, 1]

    # Save the model, results, and internal representations if output_dir is specified
    if output_dir:
        joblib.dump(model, f"{output_dir}/model.p")
        df.to_csv(f'{output_dir}/results.csv')
        np.save(f'{output_dir}/internal_representations.npy', internal_representations)

    print(results_)

    return results_, model, df



def train_homomer(
    training_data, af2_embeddings_dir: str = '../calc', C=10, balanced=0, dual=1,
    ensemble_size=1, use_pairwise=True, use_scaler=True, output_dir: str = None
) -> tuple:
    """
    Train an ensemble of logistic regression models. Same as above but predicts the specific number of subunits in a protein 

    Parameters:
        training_data (str): Path to training data.
        C (int, optional): Regularization parameter. Defaults to 10.
        balanced (int, optional): Whether to balance class weights. Defaults to 0.
        dual (int, optional): Whether to use dual formulation. Defaults to 1.
        ensemble_size (int, optional): Number of ensemble models. Defaults to 1.
        use_pairwise (bool, optional): Whether to use pairwise embeddings. Defaults to True.
        use_scaler (bool, optional): Whether to use data scaling. Defaults to True.
        output_dir (str, optional): Directory to save the output files. Defaults to None.

    Returns:
        tuple: Dictionary containing training results, trained models, and DataFrame with results.
    """

    # Load the training data
    df = pd.read_csv(training_data, sep="\t")
    # Create a state column based on the number of chains (dimer, trimer, ..., dodecamer)
    df['state'] = df['chains'] - 2  # Assuming chains start from 2 (dimer) to 12 (dodecamer)

    # Check for any monomers in the training data
    if (df['state'] == -1).any():
        raise ValueError("Training data contains monomers (state of 1). Please provide data with only homomers.")

    le = LabelEncoder()
    df['y'] = le.fit_transform(df['state'])

    # Initialize arrays to store results and internal representations
    num_classes = 11  # Dimer to dodecamer
    results = np.zeros((ensemble_size, 5, len(df), num_classes))
    internal_representations = np.zeros((ensemble_size, 5, len(df), num_classes))
    model = {}
    total_iterations = ensemble_size * 5
    iteration = 0

    # Loop over the ensemble size
    for j in range(ensemble_size):
        # Loop over the 5 AF2 models
        for i in range(5):
            iteration += 1
            print(f"Training progress: {iteration}/{total_iterations} iterations completed.")
            # Get AF2 embeddings for each PDB ID
            X = np.asarray([get_af2_emb(af2_embeddings_dir, id_=id_, model_id=i, use_pairwise=use_pairwise) for id_ in df['pdb']])
            y = df['y'].values
            # Initialize KFold cross-validation
            cv = KFold(n_splits=5, shuffle=True)

            # Loop over the cross-validation splits
            for k, (tr_idx, te_idx) in enumerate(cv.split(X, y)):
                X_tr, X_te = X[tr_idx], X[te_idx]
                y_tr, y_te = y[tr_idx], y[te_idx]

                # Scale the data if use_scaler is True
                if use_scaler:
                    sc = StandardScaler()
                    X_tr = sc.fit_transform(X_tr)
                    X_te = sc.transform(X_te)
                    model[f"scaler_{j}_{i}_{k}"] = sc

                # Initialize and train the logistic regression model
                clf = LogisticRegression(C=C, max_iter=1000, solver='liblinear',
                                         dual=False if dual == 0 else True,
                                         class_weight='balanced' if balanced == 1 else None)
                clf.fit(X_tr, y_tr)
                # Store the predicted probabilities and internal representations
                results[j, i, te_idx, :] = clf.predict_proba(X_te)
                internal_representations[j, i, te_idx, :] = clf.decision_function(X_te)
                model[f"clf_{j}_{i}_{k}"] = clf

    # Calculate the final predictions and evaluation metrics
    y_pred_bin = results.mean(axis=0).mean(axis=0).argmax(axis=1)
    results_ = {}
    results_["accuracy"] = accuracy_score(y, y_pred_bin)
    results_["f1"] = f1_score(y, y_pred_bin, average='macro')
    df["y_pred"] = y_pred_bin
    for class_idx in range(num_classes):
        df[f"prob_class_{class_idx + 2}"] = results.mean(axis=0).mean(axis=0)[:, class_idx]

    # Save the model, results, and internal representations if output_dir is specified
    if output_dir:
        joblib.dump(model, f"{output_dir}/model.p")
        df.to_csv(f'{output_dir}/results.csv')
        np.save(f'{output_dir}/internal_representations.npy', internal_representations)

    print(results_)

    return results_, model, df


def train_homomer_ordinal(
    training_data, af2_embeddings_dir: str = '../calc', C=10, balanced=0, dual=1,
    ensemble_size=1, use_pairwise=True, use_scaler=True, output_dir: str = None
) -> tuple:
    """
    Train an ensemble of logistic regression models. Same as above but predicts the specific number of subunits in a protein uses ordinal regression

    Parameters:
        training_data (str): Path to training data.
        C (int, optional): Regularization parameter. Defaults to 10.
        balanced (int, optional): Whether to balance class weights. Defaults to 0.
        dual (int, optional): Whether to use dual formulation. Defaults to 1.
        ensemble_size (int, optional): Number of ensemble models. Defaults to 1.
        use_pairwise (bool, optional): Whether to use pairwise embeddings. Defaults to True.
        use_scaler (bool, optional): Whether to use data scaling. Defaults to True.
        output_dir (str, optional): Directory to save the output files. Defaults to None.

    Returns:
        tuple: Dictionary containing training results, trained models, and DataFrame with results.
    """

    # Load the training data
    df = pd.read_csv(training_data, sep="\t")
    # Create a state column based on the number of chains (dimer, trimer, ..., dodecamer)
    df['state'] = df['chains'] - 2  # Assuming chains start from 2 (dimer) to 12 (dodecamer)

    # Check for any monomers in the training data
    if (df['state'] == -1).any():
        raise ValueError("Training data contains monomers (state of 1). Please provide data with only homomers.")

    le = LabelEncoder()
    df['y'] = le.fit_transform(df['state'])

    # Initialize arrays to store results and internal representations
    num_classes = 11  # Dimer to dodecamer
    results = np.zeros((ensemble_size, 5, len(df), num_classes))
    internal_representations = np.zeros((ensemble_size, 5, len(df), num_classes))
    model = {}
    total_iterations = ensemble_size * 5
    iteration = 0

    # Loop over the ensemble size
    for j in range(ensemble_size):
        # Loop over the 5 AF2 models
        for i in range(5):
            iteration += 1
            print(f"Training progress: {iteration}/{total_iterations} iterations completed.")
            # Get AF2 embeddings for each PDB ID
            X = np.asarray([get_af2_emb(af2_embeddings_dir, id_=id_, model_id=i, use_pairwise=use_pairwise) for id_ in df['pdb']])
            y = df['y'].values
            # Initialize KFold cross-validation
            cv = KFold(n_splits=5, shuffle=True)

            # Loop over the cross-validation splits
            for k, (tr_idx, te_idx) in enumerate(cv.split(X, y)):
                X_tr, X_te = X[tr_idx], X[te_idx]
                y_tr, y_te = y[tr_idx], y[te_idx]

                # Scale the data if use_scaler is True
                if use_scaler:
                    sc = StandardScaler()
                    X_tr = sc.fit_transform(X_tr)
                    X_te = sc.transform(X_te)
                    model[f"scaler_{j}_{i}_{k}"] = sc

                # Initialize and train the ordinal logistic regression model
                clf = mord.LogisticAT(alpha=C)
                clf.fit(X_tr, y_tr)
                # Store the predicted probabilities and internal representations
                results[j, i, te_idx, :] = clf.predict_proba(X_te)
                internal_representations[j, i, te_idx, :] = clf.decision_function(X_te)
                model[f"clf_{j}_{i}_{k}"] = clf

    # Calculate the final predictions and evaluation metrics
    y_pred_bin = results.mean(axis=0).mean(axis=0).argmax(axis=1)
    results_ = {}
    results_["accuracy"] = accuracy_score(y, y_pred_bin)
    results_["f1"] = f1_score(y, y_pred_bin, average='macro')
    df["y_pred"] = y_pred_bin
    for class_idx in range(num_classes):
        df[f"prob_class_{class_idx + 2}"] = results.mean(axis=0).mean(axis=0)[:, class_idx]

    # Save the model, results, and internal representations if output_dir is specified
    if output_dir:
        joblib.dump(model, f"{output_dir}/model.p")
        df.to_csv(f'{output_dir}/results.csv')
        np.save(f'{output_dir}/internal_representations.npy', internal_representations)

    print(results_)

    return results_, model, df


@click.command()
@click.option('--af2_embeddings_dir', type=str, default='../calc', help='Directory containing AF2 embeddings')
@click.option('--kind', type=click.Choice(['monomer', 'homomer', 'ordinal_homomer']), default='monomer', help='Type of training (monomer, homomer, or ordinal_homomer)')
@click.option('--regularization', type=float, default=10, help='Regularization parameter')
@click.option('--dual', type=int, default=1, help='Whether to use dual formulation')
@click.option('--balanced', type=int, default=0, help='Whether to balance class weights')
@click.option('--ensemble_size', type=int, default=1, help='Number of ensemble models')
@click.option('--use_scaler', type=int, default=1, help='Whether to use data scaling')
@click.option('--use_pairwise', type=int, default=1, help='Whether to use pairwise embeddings')
@click.option('--output_dir', type=str, default=None, help='Directory to save the output files')
@click.option('--training_data', type=str, default=None, help='Path to training data')
def main(af2_embeddings_dir, kind, regularization, dual, balanced, ensemble_size, use_scaler, use_pairwise, output_dir, training_data):
    try:
        print("Starting the training process...")
        if kind == 'monomer':
            results, model, df = train_monomer(training_data, af2_embeddings_dir, regularization, balanced, dual,
                                               ensemble_size, use_pairwise, use_scaler, output_dir)
        elif kind == 'homomer':
            results, model, df = train_homomer(training_data, af2_embeddings_dir, regularization, balanced, dual,
                                               ensemble_size, use_pairwise, use_scaler, output_dir)
        elif kind == 'ordinal_homomer':
            results, model, df = train_homomer_ordinal(training_data, af2_embeddings_dir, regularization, balanced, dual,
                                                       ensemble_size, use_pairwise, use_scaler, output_dir)
        print("Training completed successfully.")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
