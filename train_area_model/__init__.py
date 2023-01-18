import datetime
import logging
import os

from azureml.core import Environment, Experiment, ScriptRunConfig, Workspace
from azureml.core.authentication import ServicePrincipalAuthentication
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException

import azure.functions as func

azure_resource_group = os.getenv('AZURE_RESOURCE_GROUP')
azure_subscription_id = os.getenv('AZURE_SUBSCRIPTION_ID')
azure_ml_enviroment = os.getenv('AZURE_ML_ENVIROMENT')
azure_ml_experiment = os.getenv('AZURE_ML_EXPERIMENT')
azure_ml_workspace = os.getenv('AZURE_ML_WORKSPACE')

azure_tenant_id = os.getenv('AZURE_TENANT_ID')
service_principal_id = os.getenv('AZURE_SP_ID')
service_principal_password = os.getenv('AZURE_SP_PASSWORD')


def get_compute_target(ws, cluster_name, compute_config):
    try:
        compute_target = ComputeTarget(workspace=ws, name=cluster_name)
        logging.info('Found existing compute target.')
    except ComputeTargetException:
        logging.info('Creating a new compute target...')
        compute_config = AmlCompute.provisioning_configuration(
            vm_size='STANDARD_NC6',
            max_nodes=4
        )

        compute_target = ComputeTarget.create(ws, cluster_name, compute_config)

        compute_target.wait_for_completion(show_output=True)
    return compute_target


def main(mytimer: func.TimerRequest) -> None:
    utc_timestamp = datetime.datetime.utcnow().replace(
        tzinfo=datetime.timezone.utc).isoformat()

    logging.info('Starting training at %s', utc_timestamp)

    try:
        sp = ServicePrincipalAuthentication(tenant_id=azure_tenant_id, # tenantID
                                    service_principal_id=service_principal_id, # clientId
                                    service_principal_password=service_principal_password) # clientSecret

        ws = Workspace(azure_subscription_id, azure_resource_group, azure_ml_workspace, auth=sp)
        env = Environment(name=azure_ml_enviroment, workspace=ws)

        cluster_name = "gpu-cluster"
        compute_config = AmlCompute.provisioning_configuration(
                    vm_size='STANDARD_NC6',
                    max_nodes=4)

        compute_target = get_compute_target(ws, cluster_name, compute_config)
        logging.info(compute_target.get_status().serialize())
        
        src = ScriptRunConfig(
            source_directory='train_area_model',
            script='train_area_model.py',
            arguments=[
                '--topic-clf-name', 'topics-clf',
                '--datastore', 'postgres_decideml_pre_2'
            ],
            compute_target=compute_target,
            environment=env
        )
        
        run = Experiment(ws, azure_ml_experiment).submit(src)
    except:
        logging.exception("Could not execute experiment..")