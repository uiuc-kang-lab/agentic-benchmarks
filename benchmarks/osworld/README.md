# OSWorld Case Study

This is an analysis of issues in the [OSWorld](https://github.com/xlang-ai/OSWorld) benchmark.

## Issue Description

We find that in the OSWorld `chrome` section, many of the problems are broken due to changes made to the layout, URLs, and functionality of websites since the initial creation of the benchmark.

This is because many evals rely on HTML element selectors, such as classes and XPaths.
As time goes on and websites change their layouts, this results in

We inspected the [UI TARS](https://github.com/bytedance/UI-TARS) submission and report the impact of this issue,
since it's a fairly recent submission and is the leading entry on the leaderboard (at the time of writing).

## Examples

### HTML layout changes to `delta.com`

In sample `fc6d8143-9452-4171-9459-7f515143419a`, the AI is asked to do the following:

```
Find the status of tomorrow flights from New York airports to Columbus in Ohio.
```

Here, UI-TARS manages to navigate the `delta.com` website, and find flights from NYC to CMH on the correct date.
Visual inspection of the traces clearly shows that this should be marked as a correct submission.

However, the evaluation function looks for a particular HTML tag with the class `search-segment-cities`,
and then retrieves the text contents of this HTML element.
Due to changes in the `delta.com` website, this class name no longer exists, and so the evaluator mistakenly
assumes that the AI has arrived at the wrong webpage.

### timeslot no longer available

In sample `da46d875-6b82-4681-9284-653b0c7ae241`, the task is to book an in-person appointment for a transportation pass
on the Massachusetts Bay Transportation Authority's website.
The AI is asked to book a 10:15am slot on the first Monday in four months from the current date.

However, it seems that the website does not always have 10:15am slots.
UI-TARS correctly identified that there was no 10:15am slot available, and opted
to book a slightly different time slot.
The evaluator function looks specifically for an HTML element with the text "10:15 am",
so the problem was marked as incorrect,
when in fact a less rigid evaluation metric should have been used
(e.g. allowing any timeslot to be chosen, or using an LLM-based judge).


## Quantitative Analysis

We manually inspected the 46 `chrome` section problems of the [UI TARS](https://github.com/bytedance/UI-TARS) submission,
and found that 13 problems are affected by this issue.

We find that this issue yields an under-approximation:
for the `chrome` section, UI-TARS achieves a score of 24 / 46 before fixing evaluation criteria,
and 37 / 46 after (52% -> 80% accuracy).

Overall, this would bring UI-TARS's submission from 42.5% to 46% across the entire evaluation (7.6% relative error).

## Mitigation and Discussion

We provide patches for the problems which fix these issues.

However, this is only a temporary patch, as websites will likely continue to change over time.
A more robust solution over time is to include unit tests in the evaluation harness,
which will ensure that any changes to 

For example, a dummy agent that simply records+replays a human's motions can be used.
If the same motions no longer work due to drift in the website's layout, the dummy agent will fail
and the unit test will flag that the problem needs to be updated.

Additionally, it's possible to create more flexible evaluation criteria based around the
[web accessibility tree](https://web.dev/articles/the-accessibility-tree)
and LLM-based judges with narrowly defined criteria.
This can account for variability in the layouts and structures of websites,
and would require less maintenance cost in the future.

Moreover, to avoid maintenance costs altogether,
it might be a good idea to cache all the webpages needed for the tasks,
so that the contents and layouts of the webpages never drift.
This reduces the realism of the evaluation and might be a bit complex to implement,
but would reduce the uncertainty of incorrect samples.
