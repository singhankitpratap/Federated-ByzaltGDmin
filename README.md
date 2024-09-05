# Federated-ByzaltGDmin

# Setup on AWS and Local Machine

This guide provides detailed instructions to set up a Docker Swarm environment on AWS using EC2 instances and also locally on Docker Desktop. The setup includes one master node and multiple client nodes.

## Table of Contents
1. [AWS Setup](#aws-setup)
   - [Create Security Group](#create-security-group)
   - [Launch EC2 Instances](#launch-ec2-instances)
   - [Master Node Setup](#master-node-setup)
   - [Client Node Setup](#client-node-setup)
2. [Local Machine Setup](#local-machine-setup)
   - [Generate Data](#generate-data)
   - [Docker Swarm Setup](#docker-swarm-setup)
   - [End Simulation](#end-simulation)

---

## AWS Setup

### Create Security Group

Create a security group in AWS with the following inbound and outbound rules:

#### **Inbound Rules**:
1. **Port 22 (SSH)**:
   - **Type**: SSH
   - **Protocol**: TCP
   - **Port Range**: 22
   - **Source**: Your IP or `0.0.0.0/0` (not recommended for production)
   - Allows SSH access to manage your EC2 instances.

2. **Port 2377 (Docker Swarm Management)**:
   - **Type**: Custom TCP Rule
   - **Protocol**: TCP
   - **Port Range**: 2377
   - **Source**: Security Group ID (for internal communication)
   - Used for Docker Swarm management between nodes.

3. **Port 7946 (Node Communication - TCP)**:
   - **Type**: Custom TCP Rule
   - **Protocol**: TCP
   - **Port Range**: 7946
   - **Source**: Security Group ID (for internal communication)
   - Allows TCP communication between nodes.

4. **Port 7946 (Node Communication - UDP)**:
   - **Type**: Custom UDP Rule
   - **Protocol**: UDP
   - **Port Range**: 7946
   - **Source**: Security Group ID (for internal communication)
   - Allows UDP communication between nodes.

5. **Port 4789 (Overlay Network Traffic)**:
   - **Type**: Custom UDP Rule
   - **Protocol**: UDP
   - **Port Range**: 4789
   - **Source**: Security Group ID
   - Used for Docker Swarm's overlay network traffic.

6. **Port 80 (HTTP - Optional)**:
   - **Type**: HTTP
   - **Protocol**: TCP
   - **Port Range**: 80
   - **Source**: `0.0.0.0/0`
   - Allows HTTP traffic to services if required.

7. **Port 443 (HTTPS - Optional)**:
   - **Type**: HTTPS
   - **Protocol**: TCP
   - **Port Range**: 443
   - **Source**: `0.0.0.0/0`
   - Allows HTTPS traffic to services if required.

#### **Outbound Rules**:
- Allow all outbound traffic (default setting).

---

### Launch EC2 Instances

1. **Launch L+1 EC2 Instances**:
   - Use Amazon Linux as the operating system and assign the security group created above.

2. **Add the following User Data** when launching the instances to set up Docker and Docker Compose automatically:
   ```bash
   #!/bin/bash
   # Update the system
   sudo yum update -y
   # Install Docker
   sudo yum install -y docker
   # Start Docker service
   sudo service docker start
   # Add ec2-user to the docker group
   sudo usermod -a -G docker ec2-user
   # Enable Docker to start on boot
   sudo chkconfig docker on
   # Install Docker Compose
   sudo yum install -y python3
   sudo pip3 install docker-compose
   ```
### Master Node Setup

1. **SSH into the Master Node**:
   Use the following command to SSH into your master EC2 instance:
   ```bash
   ssh -i /path/to/your-key.pem ec2-user@<MASTER_NODE_PUBLIC_IP>
   ```
2. **Transfer the master folder, docker-compose.yml**:
   Use `scp` to transfer the necessary files from your local machine to the master node:
   ```bash
   scp -i /path/to/your-key.pem -r ./master ./docker-compose.yml ./run.py ec2-user@<MASTER_NODE_PUBLIC_IP>:/home/ec2-user/
   ```
   *Note: You can first generate data on local machine and then transfer to EC2 instance or generate the data on EC2 instance itself using* `data.py`.
   
4. **Build the Docker image**:
   Build the `master-node` Docker image from the `master` folder:
   ```bash
   docker build -t master-node ./master --rm
   ```
5. **Initialize Docker Swarm**:
   ```bash
   docker swarm init
   ```
   This command will output a docker swarm join command that you will use to join the worker/client nodes to the Swarm. It looks something like:
   ```bash
   docker swarm join --token <SWARM_TOKEN> <MASTER_NODE_PRIVATE_IP>:2377
   ```
7. **Deploy the Docker stack**:
   ```bash
   docker stack deploy -c docker-compose.yml my_stack
   ```
8. **Run the Python script**:
   ```bash
   python run.py

   ```
### Client Node Setup

1. **SSH into Each Client Node**:
   Use the following command to SSH into each client EC2 instance:
   ```bash
   ssh -i /path/to/your-key.pem ec2-user@<CLIENT_NODE_PUBLIC_IP>
    ```
2. **Transfer the client folder and data to each client node**:
   Use `scp` to transfer the necessary files from your local machine to each client node:
   ```bash
   scp -i /path/to/your-key.pem -r ./client  ec2-user@<CLIENT_NODE_PUBLIC_IP>:/home/ec2-user/
   ```
   *Note: You can first generate data on local machine and then transfer to EC2 instance or generate the data on EC2 instance itself using* `data.py`.
   
4. **Build the Docker image**:
   Build the `client-node` Docker image from the `client` folder:
   ```bash
   docker build -t client-node ./client --rm
   ```
5. **Join the Docker Swarm**:
   Use the join token from the master node to join the Swarm. The token is obtained when you initialized the Swarm on the master node:
   ```bash
   docker swarm join --token <SWARM_TOKEN> <MASTER_NODE_PRIVATE_IP>:2377
   ```
## Local Machine Setup

### Generate Data

- **Generate data locally** by running the Python script:
   ```bash
   python data.py
   ```
### Docker Swarm Setup
- Build Docker images for both the master and client nodes:
   ```bash
   docker build -t master-node ./master --rm
   docker build -t client-node ./client --rm
   ```
- Initialize Docker Swarm:
  ```bash
  docker swarm init
  ```
- Deploy the Docker stack:
    ```bash
    docker stack deploy -c docker-compose.yml my_stack
    ```
- Run the Python script to start the client simulation:
     ```bash
    python run.py
    ```
### End Simulation
   - Once the simulation is complete, leave the Docker Swarm environment:
     ```bash
     docker swarm leave --force
     ```
