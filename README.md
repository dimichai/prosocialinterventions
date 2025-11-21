# Code for paper "Can We Fix Social Media? Testing Prosocial Interventions using Generative Social Simulation"

## How to run?

1. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
2. Fill in the necessary API keys in a `.env` file:
   ```plaintext
   OPENAI_API_KEY=YOUR_KEY
   PERSPECTIVE_API_KEY=YOUR_KEY
   ```
3. Edit the main script to set the size of the simulation, the number of steps and strategies in the call to the function `run_simulation` in `main.py`:
   ```python
   run_simulation(simulation_size=500, simulation_steps=10000, 
                       user_link_strategy="on_repost_bio", 
                       timeline_select_strategy="other_partisan",
                       show_info=True, run_nr=i)
   ```
4. Run the main script:
   ```bash
   python main.py > output.txt
   ```

Outputs will be saved in results folder - a Pickle file with the whole platform state and a JSON file with results.

--------

MIT License

Copyright (c) 2025, Maik Larooij, Petter Tornberg

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
