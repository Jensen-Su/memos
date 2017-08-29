## Distill the Knowledge in a Neural Network

The main idea is to raise the temperature of the final softmax util the cumbersome model produces a suitably soft set of targets. Then to use the same high temperature when training the small model to match these soft targets.

### Logits and soft targets
Neural network typically produce class probabilities by using a "softmax" output layer that converts the **logit**, $z_i$, computed for each class into a probability, $q_i$, by comparing $z_i$ with the other logits.

$$ q_i = \frac{exp(z_i/T)}{\sum_jexp(z_j/T)} $$

where T is a temperature that is normally set to 1. Using a higher value for T produces a softer probability distribution over classes.

<font color=red> **Implement the Softmax With Temperature** </font>

```prototxt
layer {
    name: "temperature"
    type: "Scale"
    bottom: "InnerProduct1"
    top: "logits"
    param {
        lr_mult: 0  ## for teacher net
    }
    scale_param{
        filler {
            type: "constant"
            value: 0.5
        }
    }
    propagate_down: 0 ## for teacher net
}

layer {
 name: "softmax"
 type: "Softmax"
 bottom: "logits"
 top: "soft_targets"  ## or probabilities for student net
 }
```

### KD-Loss

...A better way is to simply use a weighted average of two different objective functions. 

The first objective function is the cross entropy with the soft targets and this cross entropy is computed using **the same temperature** in the softmax of the distilled model as was used for generating the soft targets from the cumbersome model.

The second objective function is the cross entropy with the correct labels. This is computed using exactly the same logits in softmax of the distilled model but at a temperature of 1.

We found that the best results were generally obtained by using a considerably lower weight on the second objective function.


<font color="red"> **Implement the KD-LOSS** </font>

```prototxt
layer {
    name: "softloss"
    type: "SigmoidCrossEntropyLoss"
    bottom: "student_prob"   ## probabilities from student net 
    bottom: "soft_targets"   ## soft targets from teacher net
    top: "cross_entropy_loss"
    propagate_down: 1
    propagate_down: 0
    loss_weight: 1.0
}
```    

