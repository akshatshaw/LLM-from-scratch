import torch
import tiktoken
# tokenizer = tiktoken.get_encoding("gpt2")

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0) # Add batch dimension
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)   # Removes batch dimension
    return tokenizer.decode(flat.tolist())

def calc_loss_batch(input_batch, target_batch, model, device):
 input_batch = input_batch.to(device) 
 target_batch = target_batch.to(device) 
 logits = model(input_batch)
 loss = torch.nn.functional.cross_entropy(
    logits.flatten(0, 1), target_batch.flatten()
 )
 return loss

def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader) 
    else:
        num_batches = min(num_batches, len(data_loader)) 
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(
                input_batch, target_batch, model, device
            )
            total_loss += loss.item() 
        else:
            break
    return total_loss / num_batches # Averages the loss over all batches

def train_model_simple(model, train_loader, val_loader,optimizer, device, num_epochs,eval_freq, eval_iter, start_context, tokenizer):
    train_losses, val_losses, track_tokens_seen = [], [], []   # tracking losses and token seen
    tokens_seen, global_step = 0, -1
    for epoch in range(num_epochs): # Main training loop
        model.train()
        for input_batch, target_batch in train_loader: # this loop iterates in batches
            optimizer.zero_grad()           # reset loss gradient from previous batch iteration
            loss = calc_loss_batch(
                 input_batch, target_batch, model, device
            )
            loss.backward() # loss gradient
            optimizer.step() # update model weights using loss gradients
            tokens_seen += input_batch.numel()
            global_step += 1
            if global_step % eval_freq == 0: # Optional evaluation
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                f"Train loss {train_loss:.3f}, "
                f"Val loss {val_loss:.3f}")
        generate_and_print_sample(model, tokenizer, device, start_context)
    return train_losses, val_losses, track_tokens_seen

# Evaluating the validation losses
def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()     # dropouts are disabled
    with torch.no_grad(): # disablibng the graddient tracking
        train_loss = calc_loss_loader(
                    train_loader, model, device, num_batches=eval_iter
            )
        val_loss = calc_loss_loader(
                     val_loader, model, device, num_batches=eval_iter
            )
    model.train()
    return train_loss, val_loss


def generate_text_simple(model, idx, max_new_tokens, context_size): 
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:] 
        with torch.no_grad():
            logits = model(idx_cond)
        
        logits = logits[:, -1, :]
        probas = torch.softmax(logits, dim=-1) 
        idx_next = torch.argmax(probas, dim=-1, keepdim=True) 
        idx = torch.cat((idx, idx_next), dim=1)
    return idx
# Generating and printing a sample text using the genearte_and_print_sample function
def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded,
            max_new_tokens=50, context_size=context_size
            )
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    txt = decoded_text.replace("\n", " ")
    print(txt) 
    model.train()
    return txt
    
