# MCP Inspector Scripts

Each JSON file in this folder describes a single tool invocation that you can load
with the Model Context Protocol Inspector. To exercise a scenario:

```bash
source .venv/bin/activate
fastmcp dev src/servers/free_tier/server.py
```

Once the Inspector opens, choose **Load Script** and select one of the JSON files.
The `arguments` block is passed directly to the tool, letting you quickly test the
success, validation error, and provider failure paths for each endpoint.
