`ps aux | grep ollama`: check if running
- The command consists of two parts, connected by a pipe (|):
ps aux: This command lists all running processes on the system.
a: Shows processes for all users.
u: Displays detailed, user-oriented output.
x: Includes processes that are not attached to a specific terminal.
grep ollama: The grep command filters the output from ps aux, searching for any lines that contain the text "ollama". This is how you find the process related to the Ollama service. 


`pkill -f ollama`: stop server