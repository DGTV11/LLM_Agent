## TODO
1) Pressing issues 
- Stop overwhelming the LLM with all that thought stuff (revert to usual, maybe support models like r1)
    - NOTE: WE GOT https://ollama.com/blog/structured-outputs 🎉
- Make memory management more modular (including main system instructions+moving function schemas that are attached to memory modules)
- Rewrite API using FastAPI library and
- Allow user to pick which function sets are wanted
2) Essential stuff
- Refine function schema paging 
- allow LLM_Agent to speak to multiple users at once (using semaphores)
3) Other goodies
- Allow it to use function-calling to search the web (selenum?)
- https://www.superagent.sh/blog/reag-reasoning-augmented-generation
- https://github.com/codelion/optillm?tab=readme-ov-file
