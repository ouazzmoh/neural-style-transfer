import torch


def gram_matrix(tensor):    
    _, d, h, w = tensor.size()
    
    # reshape so we're multiplying the features for each channel
    tensor = tensor.view(d, h * w)
    # calculate the gram matrix
    gram = torch.mm(tensor, tensor.t())
    
    return gram 


def get_features(image, model, layers_names):
    features = []
    curr_output = image
    # model._modules is a dictionary holding each module in the model
    for name, layer in model._modules.items():
        curr_output = layer(curr_output)
        if name in layers_names:
            features.append(curr_output)
    return features


def nst_step(target_image, model, content_layers, style_layers, content_features, style_features,
             content_weight, style_weight, opt):
    #Minimizing content loss 
    content_targets = get_features(target_image, model, content_layers)
    content_loss = 0
    for i in range(len(content_layers)):
        content_loss += torch.mean((content_features[i] - content_targets[i])**2)
    #Minimizing style loss
    style_targets = get_features(target_image, model, style_layers)
    style_targets = [gram_matrix(target) for target in style_targets]
    style_loss = 0
    for i in range(len(style_layers)):
        style_loss += torch.mean((style_features[i] - style_targets[i])**2)
    loss = content_weight * content_loss + style_weight * style_loss
    #Updating target_image
    opt.zero_grad() #resetting gradients because pytorch accumulates them
    loss.backward()
    opt.step()