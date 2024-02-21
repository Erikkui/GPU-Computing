# Based on PyTorch Docs

# Warm up by running all operations once

# Setting up the measurement
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)
start_event.record()

# Warmed-up operations to be timed

# Calling the end event and waiting for all tasks to complete
end_event.record()
torch.cuda.synchronize()

elapsed_time_ms = start_event.elapsed_time(end_event)
