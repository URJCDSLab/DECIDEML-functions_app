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


def main(mytimer: func.TimerRequest) -> None:
    utc_timestamp = datetime.datetime.utcnow().replace(
        tzinfo=datetime.timezone.utc).isoformat()

    logging.info('Python timer trigger function ran at %s', utc_timestamp)

    # created_at_ago = req.params.get('created_at_ago')
    created_at_ago = "24 HOURS"

    sp = ServicePrincipalAuthentication(tenant_id=azure_tenant_id, # tenantID
                                service_principal_id=service_principal_id, # clientId
                                service_principal_password=service_principal_password) # clientSecret

    ws = Workspace(azure_subscription_id, azure_resource_group, azure_ml_workspace, auth=sp)
    env = Environment.get(name=azure_ml_enviroment, workspace=ws, version='1')

    env.environment_variables = {
        "AZURE_TENANT_ID": os.getenv("AZURE_TENANT_ID"),
        "AZURE_SP_ID": os.getenv("AZURE_SP_ID"),
        "AZURE_SP_PASSWORD": os.getenv("AZURE_SP_PASSWORD"),
        "AZURE_SUBSCRIPTION_ID": os.getenv("AZURE_SUBSCRIPTION_ID"),
        "AZURE_RESOURCE_GROUP": os.getenv("AZURE_RESOURCE_GROUP"),
        "AZURE_ML_WORKSPACE": os.getenv("AZURE_ML_WORKSPACE"),
    }


    cluster_name = "gpu-cluster"

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
    logging.info(compute_target.get_status().serialize())
    
    src = ScriptRunConfig(
        source_directory='predict_job_topics',
        script='predict_job.py',
        arguments=[
            "--topics",
            '--topic-clf-name', 'topics-clf',
            '--datastore', 'postgres_decideml_pre_2',
            '--nominatim-user-agent', 'mobility@deimos-space.com',
            '--created-at-ago', created_at_ago
        ],
        compute_target=compute_target,
        environment=env
    )
    
    experiment = Experiment(ws, azure_ml_experiment)
    run = experiment.submit(src)
    logging.info(run.get_detailed_status())
    logging.info(run.get_details_with_logs())