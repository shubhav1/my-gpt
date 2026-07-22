# BPE Tokenization Experiments

## Goal

Understand the actual value of adding BPE to the baseline transformer (the first
addition made on top of it) by comparing performance without BPE against BPE at
different merge counts.

## Method

All hyperparameters held constant except merge count:

| Hyperparameter | Value |
|---|---|
| `batch_size` | 64 |
| `block_size` | 256 |
| `max_iters` | 2000 |
| `eval_interval` | 200 |
| `learning_rate` | 3e-4 |
| `eval_iters` | 200 |
| `n_embd` | 384 |
| `n_head` | 6 |
| `n_layer` | 6 |
| `dropout` | 0.2 |

Iteration count is capped at 2000 (rather than run to full convergence) as a
time/signal tradeoff since full runs take hours each on the available hardware (MPS).

**Metrics tracked:**
- Final train / validation loss
- Loss curve
- Chars-per-token (cpt) : characters / tokens on the validation set
- Wall-clock time per iteration
- Generation quality (qualitative)
- Bits-per-byte (bpb) : `val_loss / (cpt × ln(2))`

## Results summary

| Round | Merges | Final val loss | Final val bpb | cpt (val) | ms/iter | Overfit gap (val - train) |
|---|---|---|---|---|---|---|
| 1 | 0 | 1.5521 | 2.2392 | 1.0000 | ~708 | 0.2399 |
| 4 | 100 | 2.3886 | 2.1532 | 1.6005 | ~689 | 0.5655 |
| 5 | 300 | 2.9202 | 2.1634 | 1.9474 | ~691 | 0.9072 |
| 2 | 500 | 3.2116 | 2.1692 | 2.1384 | ~667 | 1.0038 |
| 3 | 1000 | 3.6472 | 2.2390 | 2.3500 | ~691 | 1.5032 |

*Ordered by merge count. Loss is not comparable across rows directly since
vocabulary changes with merge count : bpb is the fair comparison metric.*

## Key findings

**Loss curves, ascending merge count:**

| 0 merges (baseline) | 100 merges | 300 merges | 500 merges | 1000 merges |
|---|---|---|---|---|
| ![0 merge loss curve](BPE_loss_curves/0_mc.png) | ![100 merge loss curve](BPE_loss_curves/100_mc.png) | ![300 merge loss curve](BPE_loss_curves/300_mc.png) | ![500 merge loss curve](BPE_loss_curves/500_mc.png) | ![1000 merge loss curve](BPE_loss_curves/1000_mc.png) |

Here are my major takeaways:
- interesting how the graph changes
- overfitting grows with merge count (whih makes sense)
- 500 has the lowest ms/iter - what causes this? why is 1000 slower than 300? is it just noise?
- val los clearly grows with merge count - does this mean that higher merge count means more iterations are necessary for convergance or just that convergane happens at a higher loss?
- bpb is fairly stable, dipping around 100-500 merges and rising again at 100 merges.
- work in progress, this section is not done yet


## Per-round detail

### Round 1 : Baseline (0 merges)

```
chars-per-token: train 1.0000, val 1.0000
step  200: train 2.4164, val 2.4425, bpb 3.5238, 667.97 ms/iter
step  400: train 2.0936, val 2.1633, bpb 3.1209, 661.36 ms/iter
step  600: train 1.8197, val 1.9437, bpb 2.8041, 658.63 ms/iter
step  800: train 1.6602, val 1.8178, bpb 2.6226, 659.27 ms/iter
step 1000: train 1.5501, val 1.7350, bpb 2.5031, 659.10 ms/iter
step 1200: train 1.4791, val 1.6767, bpb 2.4189, 658.55 ms/iter
step 1400: train 1.4246, val 1.6344, bpb 2.3580, 701.02 ms/iter
step 1600: train 1.3781, val 1.5947, bpb 2.3007, 706.55 ms/iter
step 1800: train 1.3445, val 1.5754, bpb 2.2728, 702.58 ms/iter
step 2000: train 1.3122, val 1.5521, bpb 2.2392, 708.24 ms/iter
```

![0 merge loss curve](BPE_loss_curves/0_mc.png)

**Sample output**

```
th,
Ay in a foul to our oads along the grief,
But sevel thou for Ravein thy shat name,
And the did incham opprosic part to them.

CAMILLO:
I'll not see thee, Capulet, or where strength speach
Here with Richard's ask no the mocks died
That beggandin plains, thy hearth, there are nearfulless
Will of thy case roop for laying drack of with fair;
Whiphea's brite firge's what's in the core,
And look, onfing my broth, with spirates him.
Stave frither, dank proverain,
As cusince him wield.

Nurse:
By he
```


---

### Round 2 : 500 merges

```
chars-per-token: train 2.2327, val 2.1384
step    0: train 6.3093, val 6.2699,                1279.39 ms/iter
step  200: train 3.7914, val 3.9761, bpb 2.6825,      681.45 ms/iter
step  400: train 3.5353, val 3.7741, bpb 2.5462,      681.87 ms/iter
step  600: train 3.2121, val 3.5278, bpb 2.3797,      685.22 ms/iter
step  800: train 2.9614, val 3.3478, bpb 2.2582,      669.98 ms/iter
step 1000: train 2.7592, val 3.2536, bpb 2.1953,      664.90 ms/iter
step 1200: train 2.6090, val 3.1878, bpb 2.1509,      667.11 ms/iter
step 1400: train 2.4703, val 3.1738, bpb 2.1414,      665.68 ms/iter
step 1600: train 2.3345, val 3.1631, bpb 2.1341,      667.40 ms/iter
step 1800: train 2.2049, val 3.1903, bpb 2.1525,      667.09 ms/iter
step 2000: train 2.2078, val 3.2116, bpb 2.1692,      666.81 ms/iter
```

![500 merge loss curve](BPE_loss_curves/500_mc.png)

**Sample output (truncated in source)**

```
court is, injust of it.
Your honour intent made for time; coward finding greatness,
As Still but night shall paradise which
As cries for hate it, for no more empty dive sorrow.
GREMIO:
The word 'Of Fear and i' the eye belongs:
So, my lord. Turn my grows!' they do sound, sir;
Questaid, good Signior Baptista!
Mastera, never to acquaint Alas,
Though Paulina's love to me.
In this devise Rome, I pray thee, give me thou;
And see what keeper than thou art.
BA Place; I do could have more help
Is a truthless sms was true; for hence:
God-place will boot, and tooth all these woes,
Have I left weightness wint a roeat;
And I am crada bed, seeing more,
Than become my creaged torment's wind will bear me.
Will no more law
Deser hatered of her death,
If I must confess of a word's daughter; then your flood, throngs' handless
Since present watch word in your down,
And yet she unleeply bite danger:
And when I had so doubterer, may serve your plead,
And sever'd succlof,
I'll dally a true flies are marigal to his royal now.
DUKE OF YORK:
Can this bride the aunt of mine eyes.
DUKE OF YORK:
Yet givener, my grandam, lords, g
```


---

### Round 3 : 1000 merges

```
chars-per-token: train 2.5775, val 2.3500
step  200: train 4.0916, val 4.1997, bpb 2.5782, 701.14 ms/iter
step  400: train 3.8238, val 3.9761, bpb 2.4410, 686.26 ms/iter
step  600: train 3.5838, val 3.8123, bpb 2.3404, 679.09 ms/iter
step  800: train 3.3149, val 3.6534, bpb 2.2429, 689.73 ms/iter
step 1000: train 3.0793, val 3.5579, bpb 2.1842, 680.91 ms/iter
step 1200: train 2.8817, val 3.5344, bpb 2.1698, 675.84 ms/iter
step 1400: train 2.7074, val 3.5327, bpb 2.1688, 680.70 ms/iter
step 1600: train 2.5145, val 3.5500, bpb 2.1794, 680.29 ms/iter
step 1800: train 2.3301, val 3.5913, bpb 2.2048, 676.05 ms/iter
step 2000: train 2.1440, val 3.6472, bpb 2.2390, 691.47 ms/iter
```

Note: train loss keeps falling past step 1200 while val loss and val bpb bottom
out around step 1400 and then rise : the clearest overfitting signal of the five
runs.

![1000 merge loss curve](BPE_loss_curves/1000_mc.png)

**Sample output**

```
cares follow in this's brother being rave:
Good mistrother, if thou think'st me thy name,
Save them light, the mount of all my king,
That my queen, my daughter's here, mightial to my Verona fault,
My brother traitor and my brother.

LADY ANNE:
And you other be with rail upon my soul.

GLOUCESTER:
And speed my father, in his curses place with him!

LADY ANNE:
O, you peradful time of vengethen!
That press do not profrication uion but a pace
But that poison that adden days from hardict this suit hangman
Which for my black mile;
How fares my soul shall I fear my brother Masters sway;
In my mect's task'd my despisedness of my witness
A was prodigations,
And not my sprors mantairs are returned in your fate:
For me, I do not lose my soul to thee;
Four couch me with some released me travel,
If you treason say I strip, you found beat unto up the part of you,
In you lesson up by the hand: you indept to your emboy;
Your foul wrong, you shall as fall for caveNow Paulara;
Which traitors in pure and paper and cried bitter high,
As you or thy sun, utterance, but to me to prevail.

KING RICHARD II:
Fie, fair vow of befalled; and with praised escap in charge
This hand are nails shallow with strily-wise or escraft of blood.

HENRY BOLINGBROKE:
Sweet Gentle the king at sin thousand seld,
And all with a busation of thy royalty,
HNORTHUMBERLAND:
```


---

### Round 4 : 100 merges

```
chars-per-token: train 1.6218, val 1.6005
step  200: train 3.2294, val 3.3267, bpb 2.9988, 691.74 ms/iter
step  400: train 2.8795, val 3.0546, bpb 2.7535, 673.47 ms/iter
step  600: train 2.5311, val 2.7826, bpb 2.5083, 682.92 ms/iter
step  800: train 2.3213, val 2.6096, bpb 2.3523, 685.24 ms/iter
step 1000: train 2.1957, val 2.5252, bpb 2.2763, 702.84 ms/iter
step 1200: train 2.0925, val 2.4733, bpb 2.2295, 690.22 ms/iter
step 1400: train 2.0138, val 2.4232, bpb 2.1843, 696.91 ms/iter
step 1600: train 1.9394, val 2.4147, bpb 2.1766, 685.89 ms/iter
step 1800: train 1.8823, val 2.3951, bpb 2.1590, 685.29 ms/iter
step 2000: train 1.8231, val 2.3886, bpb 2.1532, 689.26 ms/iter
```

![100 merge loss curve](BPE_loss_curves/100_mc.png)

**Sample output**

```
comfort
Is fell unrestaught from, I think with her stewarning oons.
Lords, then.

ISABELLA:
Will you be sly well, to the doing in seat,
And she shall keeps to Romeo.

DUKE VINCENTIO:
Both not you this, like to sive; but thou i' of a
deed, holy were foul deniam it: quick them not are, thou
while we stay with a hearing, shall to die at the number:
A poison of crothechether me in
By mine eye oval and turn thee at my wit, I. You
shepe--which next to him that, come to my broth:
beard-druck not taken. Like the babe stay allower,
a crease: had upon my neck feed entable do blam, and
fore him again alone betternock restrain one do high
came us to thee: this is this should dissember it: a
lover I dog thee would tipel of brought out him; they leave their
dole-buke an oath; easy for them: but if you say boath,
the weak: put it wears, remedy raspe
```


---

### Round 5 : 300 merges

```
chars-per-token: train 2.0096, val 1.9474
step  200: train 3.5977, val 3.7256, bpb 2.7601, 687.81 ms/iter
step  400: train 3.3215, val 3.5171, bpb 2.6056, 665.62 ms/iter
step  600: train 2.9630, val 3.2480, bpb 2.4062, 662.85 ms/iter
step  800: train 2.7189, val 3.0838, bpb 2.2846, 668.76 ms/iter
step 1000: train 2.5547, val 2.9797, bpb 2.2075, 663.76 ms/iter
step 1200: train 2.4236, val 2.9232, bpb 2.1656, 671.04 ms/iter
step 1400: train 2.3083, val 2.9101, bpb 2.1559, 691.32 ms/iter
step 1600: train 2.2098, val 2.9006, bpb 2.1488, 689.92 ms/iter
step 1800: train 2.1074, val 2.8991, bpb 2.1477, 689.78 ms/iter
step 2000: train 2.0130, val 2.9202, bpb 2.1634, 690.82 ms/iter
```

![300 merge loss curve](BPE_loss_curves/300_mc.png)

**Sample output**

```
courseserve.

First Murderer:
Ay, a bondman, out sure
A lacks in our blemission.

Second Citizen:
Where is a noble vessel, makes honour head
Are: could the mirth glister'd.

Second Citizen:
I know it expass that you tell men the next Thou would
O, a honour, therein, to Stere you live
Citizens in a fool in this wench.

First Citizen:
Down, my lord.

CORIOLANUS:
What set you do I b dearly appeal?

CORIOLANUS:
Nay, they come bitter for your thing it; being a virgin
That, we have been in your bold fleght,
Should nothing.

CORIOLANUS:
My?

MENENIUS:
Do you sir: I tell you again. He should I had
Titus his good deny: no wanton you at your worth
Your royal complay, his news.

Second Citizen:
I'll brain thee.

COMINIUS:
You have smiling against all this proceeding
It have been i' such a speech; but it is cogg,
If it not a villain, quarrel with an icatcratage or
fold within your envent.
You he have be many hide, come to your best;
End, child, indeed you our title that the day nor
Whe
```


## Qualitative comparison of sample outputs

Loss gets worse with increasing merge count, and the quality of sample outputs 
seem to follow that trend. The baseline (0 merges) reads best by far. Within
the BPE runs, 500 merges is noticeably better than 100 and 300, and 1000 merges
is clearly the worst, with the longest and most tangled token garbling
("HNORTHUMBERLAND," "escraft"). Here's my rough ranking: 0 > 500 > 100 > 300 > 1000
- this ranking isn't conclusive since it's based on one sample + eyeballing

## Open questions
It's clear that this isn't a very accurate "test" for how useful BPE is since I held all hyperparameters constant and didn't run to convergence. 500 merges does seem to emerge as the best BPE setting, but we 
don't have a clear comparison of how it performs in comparison to the baseline when run to convergence. 

This is also an interesting thing to think about for ablation experiments in general: how do you design an
experiment that accurately compares two different models when it's not feasible to run to convergence? How do you choose the hyperparameters in the first place such that they're optimal for both models?

Regardless, for the integrity of the experiment, I want to do a REAL ablation where I run to convergence for the baseline and for 500 merges to see how BPE actually performs.

Convergence criterion: Validation bpb is evaluated every 100 iterations. Training is considered converged once three consecutive validation bpb measurements fall within a 0.01 range of each other. The reported result is the checkpoint with the lowest validation bpb observed over the run, not necessarily the final step.

### Baseline results (to convergence)
work in progress

### BPE (500 merges) results (to convergence)
work in progress