This is an analysis of issues in the [OSWorld](https://github.com/xlang-ai/OSWorld) benchmark.


# chrome

This task subset contains a handful of errors, mainly due to changes to various websites
on the internet, since OSWorld tests an agent's ability to navigate various real-world websites.

One of the techniques used in the evaluation criteria is locating elements within the
HTML dump of the active Chrome tab running within the VM. 
This is done by searching for specific classes, or XPaths.

However, over time, some of the XPaths of the target elements have drifted from their
original values,
since many of the websites used by the eval have updated their structure.

While we have provided a number of fixes for issues we found,
we also recommend unit and component tests for each evaluation that determine
if the contents of the web pages have changed, so that a maintainer can fix the evaluation
criteria of a benchmark problem.
Especially since the web is in constant flux, and the underlying websites do often change.

Additionally, we recommend more flexible evaluation criteria based around the
[web accessibility tree](https://web.dev/articles/the-accessibility-tree)
and more LLM-based judges with narrowly defined criteria.


## 2888b4e6-5b47-4b57-8bf5-c73827890774

For this sample, we find that the "Sales & Discount" section on the Macy's website is
no longer available.

Thus, we opt to drop that critera altogether.

