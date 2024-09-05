import os
import logging
import torch
import numpy as np
from flask import Flask, request, jsonify
import json
import matplotlib.pyplot as plt
from pathlib import Path
from threading import Condition
import random
from utils import subspace_median, weiszfeld_algorithm

torch.set_default_dtype(torch.float64)

app = Flask(__name__)

# Global variables
clients = {}
thresholds = {}
initializations = {}
gradients = {}
weights = None
expected_clients = 6
received_clients = 0
fetched_clients = 0
rounds = 150  # Number of federated learning rounds
current_round = 0
state = 'THRESHOLD'  # Initial state

# Initialize logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

condition = Condition()

def log_error(filename, iteration, error):
    with open(filename, 'a') as f:
        f.write(f"{iteration},{error}\n")

def plot_errors(iterations, errors):
    iterations_tensor = torch.tensor(iterations, dtype=torch.float64)
    errors_tensor = torch.tensor(errors, dtype=torch.float64)
    current = Path(".")
    file_path = current / "data"
    plt.figure()
    plt.semilogy(iterations_tensor.numpy(), errors_tensor.numpy(), '-*', color='black', markersize=10, markevery=10)
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.title('Iteration vs Error')
    plt.grid(True, which="both", ls="--")
    plt.savefig(file_path / 'error_plot.png')
    plt.close()

def read_errors(filename):
    iterations = []
    errors = []
    with open(filename, 'r') as f:
        for line in f:
            iteration, error = line.split(',')
            iterations.append(int(iteration))
            errors.append(float(error))
    return iterations, errors


# Aggregation functions
def Byz_altGDmin_threshold_server(alpha_l_list, L_byz, L):
    alpha_mean = -10 / L * sum(alpha_l_list)
    selected_indices = random.sample(range(len(alpha_l_list)), L_byz)
    for idx in selected_indices:
        alpha_l_list[idx] = alpha_mean
    alpha = np.median(alpha_l_list)
    return alpha

def Byz_altGDmin_initialization_server(U_0_list, L_byz, L, n, r, U_star, T_gm, epsilon_gm):
    alt_matrix = torch.tensor([10 if (i + j) % 2 == 0 else -10 for i in range(n) for j in range(r)], dtype=torch.float64).reshape(n, r)
    selected_indices = random.sample(range(L), L_byz)
    for idx in selected_indices:
        U_0_list[idx] = alt_matrix.float()
    U_out = subspace_median(U_0_list, T_gm, epsilon_gm)
    U_0 = U_out
    error_init = torch.norm((torch.eye(n, dtype=torch.float64) - U_star @ U_star.t()) @ U_0)
    logger.info(f"Initialization Error: {error_init}")
    return U_0

def Byz_altGDmin_GD_server(gradients, L_byz, L, eta, m, U_t_minus_1, U_star, T_gm, epsilon_gm, iteration, r):
    grad_mean = -10 / L * sum(gradients)
    selected_indices = random.sample(range(L), L_byz)
    for idx in selected_indices:
        gradients[idx] = grad_mean.to(torch.float64)
    grad_flattened = [grad.reshape(-1).to(torch.float64) for grad in gradients]
    grad_gm_flat = weiszfeld_algorithm(grad_flattened, T_gm, epsilon_gm)
    grad_gm = grad_gm_flat.view(gradients[0].shape).to(torch.float64)
    U_plus = torch.linalg.qr(U_t_minus_1 - eta * grad_gm)[0]
    U_t = U_plus
    error_gd = torch.norm((torch.eye(U_t.size(0), dtype=torch.float64) - U_star @ U_star.t()) @ U_t) / torch.sqrt(torch.tensor(r, dtype=torch.float64))
    logger.info(f"SD: {error_gd}")
    log_error("error_log.txt", iteration, error_gd)
    return U_t, error_gd

@app.route('/subscribe', methods=['GET'])
def subscribe():
    global clients
    client_id = len(clients) + 1
    clients[client_id] = None
    logger.info(f"Client {client_id} subscribed.")
    return jsonify({'id_client': client_id})

@app.route('/push-threshold', methods=['POST'])
def push_threshold():
    global thresholds, received_clients, state
    data = request.json
    client_id = data['id_client']
    alpha_l = torch.tensor(data['data'])

    with condition:
        thresholds[client_id] = alpha_l
        logger.info(f"Received threshold from client {client_id}.")
        received_clients += 1

        if received_clients == expected_clients:
            alpha_l_list = [thresholds[client_id] for client_id in thresholds]
            alpha = Byz_altGDmin_threshold_server(alpha_l_list, L_byz, expected_clients)
            logger.info(f"Calculated alpha: {alpha}")
            for client_id in clients:
                thresholds[client_id] = alpha
            state = 'INITIALIZATION'
            received_clients = 0
            condition.notify_all()

    return 'Received', 200

@app.route('/get-alpha/<int:client_id>', methods=['GET'])
def get_alpha(client_id):
    global thresholds, fetched_clients
    with condition:
        condition.wait_for(lambda: client_id in thresholds and state == 'INITIALIZATION')
        alpha = thresholds.pop(client_id)
        fetched_clients += 1
        if fetched_clients == expected_clients:
            fetched_clients = 0
            condition.notify_all()
        return jsonify(alpha.tolist())

@app.route('/push-initialization', methods=['POST'])
def push_initialization():
    global initializations, received_clients, weights, state
    data = request.json
    client_id = data['id_client']
    U_0_l = torch.tensor(data['data'])

    with condition:
        initializations[client_id] = U_0_l
        logger.info(f"Received initialization from client {client_id}.")
        received_clients += 1

        if received_clients == expected_clients:
            logger.info("All initializations received. Starting aggregation...")
            U_0_list = [initializations[client_id] for client_id in initializations]
            U_0 = Byz_altGDmin_initialization_server(U_0_list, L_byz, expected_clients, U_0_list[0].shape[0], U_0_list[0].shape[1], U_star, T_gm, epsilon_gm)
            weights = U_0
            for client_id in clients:
                initializations[client_id] = weights
            state = 'INITIALIZATION_AGGREGATED'
            received_clients = 0
            logger.info("Aggregated")
            condition.notify_all()

    return 'Received', 200

@app.route('/push-gradients', methods=['POST'])
def push_gradients():
    global gradients, received_clients, current_round, fetched_clients, weights, state
    data = request.json
    client_id = data['id_client']
    grad_f_l = torch.tensor(data['data'])

    with condition:
        gradients[client_id] = grad_f_l
        logger.info(f"Received gradients from client {client_id}.")
        received_clients += 1

        if received_clients == expected_clients:
            gradients_list = [gradients[client_id] for client_id in gradients]
            weights, _ = Byz_altGDmin_GD_server(gradients_list, L_byz, expected_clients, eta, m, weights, U_star, T_gm, epsilon_gm, current_round, r)
            current_round += 1
            logger.info(f"Round {current_round} completed.")
            for client_id in clients:
                gradients[client_id] = weights
            if current_round >= rounds:
                iterations, errors = read_errors('error_log.txt')
                plot_errors(iterations, errors)
                state = 'COMPLETED'
            else:
                state = 'AGGREGATION_DONE'
            received_clients = 0
            condition.notify_all()

    return 'Received', 200

@app.route('/get-weights/<int:client_id>', methods=['GET'])
def get_weights(client_id):
    global weights, fetched_clients, state
    with condition:
        if state == 'INITIALIZATION_AGGREGATED':
            condition.wait_for(lambda: client_id in initializations)
            weights = initializations.pop(client_id)
        else:
            condition.wait_for(lambda: client_id in gradients and state == 'AGGREGATION_DONE')
            weights = gradients.pop(client_id)
        fetched_clients += 1
        if fetched_clients == expected_clients:
            fetched_clients = 0
            state = 'GD'
            condition.notify_all()
        return jsonify(weights.tolist())

@app.route('/state', methods=['GET'])
def get_state():
    global current_round, rounds, expected_clients, received_clients, fetched_clients, state
    return jsonify({
        'current_round': current_round,
        'rounds': rounds,
        'expected_clients': expected_clients,
        'received_clients': received_clients,
        'fetched_clients': fetched_clients,
        'state': state
    })

if __name__ == '__main__':
    logger.info("Master node started.")
    current1 = Path("data")
    file_path = current1 / "U_star.pth"
    U_star = torch.load(file_path)
    with open(current1 / "scalars.json", 'r') as json_file:
        scalars = json.load(json_file)

    kappa = scalars["kappa"]
    mu = scalars["mu"]
    sigma_max = scalars["sigma_max"]
    q = scalars["q"]
    r = scalars["r"]
    L = scalars["L"]
    m = scalars["m"]
    n = scalars["n"]
    m_L = scalars["m_L"]
    eta = torch.tensor(0.5 / (m_L * sigma_max**2), dtype=torch.float64)
    L_byz = 2

    T = 150
    T_gm = 100
    epsilon_gm = 1e-5
    app.run(host='0.0.0.0', port=80)
