# Safety-ALFRED

## Quickstart
Install requirements(conda):
```bash
$ conda create -n ai2thor python==3.10
$ conda activate ai2thor
$ pip install -r requirements.txt
```
Evaluate model on single traj:
```bash
# Setup API_KEY (default is openrouter api)
$ export API_KEY="your_api_key_here"
$ python models/eval/eval_llm_astar.py --debug --traj_file=examples/pick_and_place_simple_bowl_microwave/metal_in_bowl/traj_data.json
```

## TODOs
- [ ] *Finish implementing task generation pipeline - see `gen_safety`*
- [ ] Adapt the expert traj generator and make sure that the trajs are safe & success
- [ ] Update `trace_to_ctl.py` to include all the new propositions
- [ ] Implement baseline comparisons

### Adapt expert traj generator
The legacy expert traj generator is stored under `gen/` - can either get rid off the deprecated functions to use the new ai2thor version or can start one from scratch. The goal of the generator is to be able to generate a successful & safe action sequence given a traj_data.json.

Note that if you want to modify the legacy expert traj generator, you should probably remove all the usage of gt_graph and alike. Those use external maps that are no longer applicable to the current ai2thor version. If you wish to do any env querries, ai2thor now provides `GetReachablePositions`, `GetInteractablePoses`, and other helpful functions that you can directly call. Check out the documentation on their webpage.

### Update `trace_to_ctl.py` to include all the new propositions
The current file already includes some examples for how you should go about adding nodes/edges from the traj trace. Most of the nodes should be simply added by a lookup function. The edges might be more complicated most of the times and can require some extra logic and thresholds. These nodes and edges will then be evaluated using the `safety_rules_object.json`. The updated version should allow us to evaluate all the existing rules.

### Implement `generate_task.py`
The overall pipeline is somewhat written. See `put_candle_near_flammable` as an example. We now need to extend this to all actions and all safety rules included below in the action list. You can use `safety_rules_object.json` as a reference to think about what safety rules you can use for each action. If there are some safety rules that you think is not covered, feel free to send a message in the group and we can talk about adding them!

Remember to update README as you implement so we can have a running list of safety_rules for each action (does not have to be very specific).

** Action List **
- [ ] GotoLocation (Navigation only)
    - Broken vase (or other things) on the way
    - opened door on the way
- [ ] PickupObject
    - Cannot think of anything so far
- [ ] PutObject
    - Put certain "dangerous" object at the target location first such that placing the target object there is not safe
    - Similar to metal/forbidden material in a bowl setup
- [ ] OpenObject & CloseObject
    - Put certain object in front of the target such that the open/close door trajectory would cause collision
    - Put certain objects in hand to achieve this same effect
- [ ] ToggleObjectOn & ToggleObjectOff
    - Certain object cannot be toggled on while other "dangers" are present
- [ ] (need to wait for a bit) PushObject & PullObject
    - Lookat ai2thor rearrangement challenge


### From Old README
Benchmarking:
```bash
$ bash scripts/run_all.bash
```

Safety Eval:
```bash
$ python safety_eval/ctl_full_pipeline.py   --task-name pick_and_place_simple-Kettle-None-StoveBurner-2  --constraints-json safety_rules_object.json

# OR
$ python safety_eval/ctl_full_pipeline.py --model-name openai/gpt-5

```


## Headless Server
```bash
## Setup Xvfb for AI2-THOR
# Start Xvfb on display :99
Xvfb :99 -screen 0 1024x768x24 -ac +extension GLX +extension RANDR +extension RENDER &
export DISPLAY=:99

# Check if thor works
python scrips/check_thor.py
  ###############
  ## (300, 300, 3)
  ## Everything works!!!

```
**Then change DISPLAY constant value to the screen number (99 here) in [gen/constants.py](gen/constants.py)**

Also, checkout this guide: [Setting up THOR on Google Cloud](https://medium.com/@etendue2013/how-to-run-ai2-thor-simulation-fast-with-google-cloud-platform-gcp-c9fcde213a4a)

## Citation

If you find the dataset or code useful, please cite:

```
@inproceedings{ALFRED20,
  title ={{ALFRED: A Benchmark for Interpreting Grounded
           Instructions for Everyday Tasks}},
  author={Mohit Shridhar and Jesse Thomason and Daniel Gordon and Yonatan Bisk and
          Winson Han and Roozbeh Mottaghi and Luke Zettlemoyer and Dieter Fox},
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2020},
  url  = {https://arxiv.org/abs/1912.01734}
}
```

## License

MIT License
