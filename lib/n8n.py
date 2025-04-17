import os
import json
import httpx
from typing import Dict, Any, List

def get_n8n_api_key() -> str:
    """Get the n8n API key from environment variables."""
    return os.getenv("N8N_API_KEY", "")

async def get_workflow_template(workflow_name: str) -> str:
    """Get the workflow template from the workflows directory."""
    try:
        with open(f"workflows/{workflow_name}.json", "r") as f:
            return f.read()
    except Exception as e:
        print(f"Error reading workflow template: {e}")
        raise

async def get_user_workflows(user_id: str) -> List[Dict[str, Any]]:
    """Get all workflows associated with a specific user.
    
    Args:
        user_id: The user ID to filter workflows by
        
    Returns:
        A list of workflows that contain the user's ID in their nodes
    """
    try:
        n8n_api_key = get_n8n_api_key()
        async with httpx.AsyncClient() as client:
            # Get all workflows
            response = await client.get(
                "https://n8n.subthread.studio/api/v1/workflows",
                headers={
                    "Content-Type": "application/json",
                    "X-N8N-API-KEY": n8n_api_key,
                },
            )
            
            if response.status_code != 200:
                raise Exception(f"Failed to get workflows: {response.text}")
                
            workflows = response.json()["data"]
            
            # Filter workflows that contain the user's ID
            user_workflows = []
            for workflow in workflows:
                # Check if any node contains the user ID
                for node in workflow.get("nodes", []):
                    if "parameters" in node:
                        # Convert parameters to string to search for user ID
                        params_str = json.dumps(node["parameters"])
                        if user_id in params_str:
                            user_workflows.append(workflow)
                            break
            
            return user_workflows
            
    except Exception as error:
        print(f"Error getting user workflows: {error}")
        raise

async def create_scheduled_workflow(
    cron: str,
    phone_number: str,
    user_id: str,
    message: str,
    title: str,
) -> Dict[str, Any]:
    """Create and activate a scheduled workflow in n8n.
    
    Args:
        cron: The cron expression for scheduling
        phone_number: The phone number to call
        user_id: The user ID to get profile facts for
        
    Returns:
        The workflow data from n8n
    """
    try:
        # Get the workflow template
        workflow_template = await get_workflow_template("elderly-companion")
        
        # Inject user data directly into the workflow nodes
        workflow_json = json.loads(
            workflow_template
            .replace("{{ $json.phoneNumber }}", phone_number)
            .replace("{{ $json.userId }}", user_id)
            .replace("{{ $json.cron }}", cron)
            .replace("{{ $json.ELDERLY_COMPANION_API }}", os.getenv("ELDERLY_COMPANION_API"))
            .replace("{{ $json.message }}", message)
            .replace("{{ $json.workflowName }}", title)
        )
        
        # Create the workflow in n8n
        n8n_api_key = get_n8n_api_key()
        async with httpx.AsyncClient() as client:
            # Create workflow
            response = await client.post(
                "https://n8n.subthread.studio/api/v1/workflows",
                json=workflow_json,
                headers={
                    "Content-Type": "application/json",
                    "X-N8N-API-KEY": n8n_api_key,
                },
            )
            
            if response.status_code != 200:
                raise Exception(f"Failed to create workflow: {response.text}")
                
            data = response.json()
            
            # Activate the workflow
            activate_response = await client.post(
                f"https://n8n.subthread.studio/api/v1/workflows/{data['id']}/activate",
                headers={
                    "Content-Type": "application/json",
                    "X-N8N-API-KEY": n8n_api_key,
                },
            )
            
            if activate_response.status_code != 200:
                raise Exception("Failed to activate workflow")
                
            print("Workflow activated successfully")
            return data
            
    except Exception as error:
        print(f"Error creating workflow: {error}")
        raise

async def delete_scheduled_workflow(workflow_id: str) -> None:
    """Delete a scheduled workflow from n8n.
    
    Args:
        workflow_id: The ID of the workflow to delete
        
    Raises:
        Exception: If the workflow deletion fails
    """
    try:
        n8n_api_key = get_n8n_api_key()
        async with httpx.AsyncClient() as client:
            # Delete the workflow
            response = await client.delete(
                f"https://n8n.subthread.studio/api/v1/workflows/{workflow_id}",
                headers={
                    "Content-Type": "application/json",
                    "X-N8N-API-KEY": n8n_api_key,
                },
            )
            
            if response.status_code != 200:
                raise Exception(f"Failed to delete workflow: {response.text}")
                
            print(f"Workflow {workflow_id} deleted successfully")
            
    except Exception as error:
        print(f"Error deleting workflow: {error}")
        raise 