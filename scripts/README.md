# Scripts

Utility scripts for the AI Interview Coach project.

## upload_prompts_to_langfuse.py

Upload all local prompt files to Langfuse for centralized prompt management.

### Usage

```bash
python scripts/upload_prompts_to_langfuse.py
```

### Prerequisites

1. Set up Langfuse credentials in your `.env` file:
   ```
   LANGFUSE_PUBLIC_KEY=your_public_key
   LANGFUSE_SECRET_KEY=your_secret_key
   LANGFUSE_HOST=https://cloud.langfuse.com  # or your self-hosted instance
   ```

2. Ensure all prompts exist in the `prompts/` directory

### What it does

1. Scans the `prompts/` directory for all `.txt` files
2. Reads each prompt file
3. Uploads to Langfuse with the naming convention: `{agent_type}_{prompt_name}`
4. Tags prompts with appropriate labels for organization

### Example

Local file structure:
```
prompts/
  interviewer/
    system.txt
    greeting.txt
    response.txt
  observer/
    system.txt
    analysis.txt
  feedback_generator/
    system.txt
    feedback.txt
```

Uploaded to Langfuse as:
- `interviewer_system`
- `interviewer_greeting`
- `interviewer_response`
- `observer_system`
- `observer_analysis`
- `feedback_generator_system`
- `feedback_generator_feedback`

### Benefits

- **Version Control**: Track prompt changes in Langfuse
- **A/B Testing**: Test different prompt versions
- **Collaboration**: Team members can edit prompts in Langfuse UI
- **Real-time Updates**: Update prompts without redeploying code
- **Fallback**: System still works with local files if Langfuse is unavailable
