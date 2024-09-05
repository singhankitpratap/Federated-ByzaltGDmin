import logging
import requests
import torch
import json
from pathlib import Path
import os
import time
from node import Byz_altGDmin_threshold, Byz_altGDmin_initialization, Byz_altGDmin_GD_iterations

torch.set_default_dtype(torch.float64)

MASTER_URL = 'http://master:80'
weights = None

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def subscribe():
    response = requests.get(f'{MASTER_URL}/subscribe')
    id_client = response.json()['id_client']
    logger.info(f"Subscribed to the Master Node. ID client assigned {id_client}.")
    return id_client

def load_data(client_id):
    current = Path(".")
    data_path = current / "data" / f'data_node_{client_id-1}.pth'
    data = torch.load(data_path)
    return data['Y_l'], data['A_l']

def send_data(id_client, data, endpoint):
    response = requests.post(f'{MASTER_URL}/{endpoint}', json={'id_client': id_client, 'data': data.tolist()})
    logger.info(f"Sent data to endpoint {endpoint}: {data.tolist()}")

def receive_data(client_id, endpoint):
    while True:
        response = requests.get(f'{MASTER_URL}/{endpoint}/{client_id}')
        if response.status_code == 200:
            data = torch.tensor(response.json())
            logger.info(f"Received data from endpoint {endpoint} for CLIENT_ID {client_id}.")
            return data
        logger.info(f"CLIENT_ID {client_id} waiting for data from endpoint {endpoint}...")
        time.sleep(5)

def wait_for_state(desired_state):
    while True:
        response = requests.get(f'{MASTER_URL}/state')
        if response.status_code == 200:
            state_info = response.json()
            if state_info['state'] == desired_state:
                return
        logger.info(f"Waiting for state to change to {desired_state}...")
        time.sleep(5)

if __name__ == '__main__':
    id_client = subscribe()
    Y_l, A_l = load_data(id_client)
    current = Path("data")
    with open(current / "scalars.json", 'r') as json_file:
        scalars = json.load(json_file)

    # Access the scalar values
    kappa = scalars["kappa"]
    mu = scalars["mu"]
    sigma_max = scalars["sigma_max"]
    q = scalars["q"]
    r = scalars["r"]
    L = scalars["L"]
    m = scalars["m"]
    n = scalars["n"]
    m_L = scalars["m_L"]

    # Step 1: Compute and send threshold
    alpha_l = Byz_altGDmin_threshold(Y_l, kappa, mu, m, q)
    send_data(id_client, alpha_l, 'push-threshold')
    wait_for_state('INITIALIZATION')

    # Step 2: Receive and use alpha for initialization
    alpha = receive_data(id_client, 'get-alpha')
    U_0_l = Byz_altGDmin_initialization(Y_l, A_l, alpha, r, q)
    send_data(id_client, U_0_l, 'push-initialization')
    wait_for_state('INITIALIZATION_AGGREGATED')

    # Step 3: Gradient Descent Iterations
    for _ in range(150):  # Number of federated learning rounds
        U_t_minus_1 = receive_data(id_client, 'get-weights')
        wait_for_state('GD')
        grad_f_l = Byz_altGDmin_GD_iterations(Y_l, A_l, U_t_minus_1, q)
        send_data(id_client, grad_f_l, 'push-gradients')
        logger.info(f"CLIENT_ID {id_client} sent gradients.")
        wait_for_state('AGGREGATION_DONE')
        # time.sleep(5)  # Wait before the next round
