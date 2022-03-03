# Using Git in our daily workflow

Git is an essential tool in our day-to-day coding life. Below you will find Git hints and best 
practices and conventions that we aim to follow at ML6.


## Naming conventions

* Always start development with creating a new branch.
* Name the branch according to our conventions.
* Use feature/ prefix for a new feature and `bugfix/` for fixing a bug.
* Separate the words in the branch name with dash.

##### Examples:

* `feature/docs`
* `bugfix/data-pipeline`


## Commit messages

* Write clear commit messages. A commit message should be a short summary of your changes.
* Use imperative, present tense statements (e.g. 'change', and not 
'changed' or 'changes').
* Try to limit the message to about 50 characters or less to keep it clear and readable.

#### Examples:

* `Add io module documentation`
* `Delete preprocessing step in data pipeline`


## Clean commit history

* Only commit the code you tested.
* Make sure the change is a logical chunk that is completed and has no side effects.
* Keep separate parts of your change in separate commits. For example, fixing two different bugs 
should produce two separate commits. This will make your commit history clean and structured. 
In addition, it would be easier to possibly roll back if something goes wrong, and will also help 
reviewers to understand the changes faster.
* Before merging, small commits representing intermediate steps in context of a bigger change or 
made in response to PR comments can be cleaned up by squashing them.

### Some useful Git commands

#### `git commit`

* `--amend`
Replaces the last commit done with a new one.

> Use the `--no-edit` flag if you want to keep the commit message.

> Also make sure to to check what your last commit was before amending.

* `--fixup HASH`
Creates special commit with prefix linked to the specified commit. This commit can be used in 
combination with `rebase --autosquash` to correct past commits. 

#### `git rebase`

* `--onto`
Allows the user to specify the starting point of the rebase.

* `-i`
Interactive rebase - allows the user the select which commits to rebase.

* `--autosquash`
Can only be used with the interactive rebase. Will automatically squash commits with the prefix 
`squash!` or `fixup!` with linked list while rebasing.

> Pushing to remote: don’t forget that the above commands changes the commit. This means that to push 
it to a remote branch you’ll need to use the `--force` flag or `--force-with-lease` flag.
`--force-with-lease` is a safer option than `--force` as it prevents overwriting the remote if 
additional commits have been added to the branch.

##### Example: drop commit with `git rebase -i`
1.
```bash
    C -- D -- E (feature/my-branch, HEAD)
  /
A -- B (main)
```

2. `git rebase -i main`

```bash
pick 4e37e38 commit C
# pick c78980a commit D
pick fbcfda8 commit E
```

3.
```bash
           C’ -- E’ (feature/my-branch, HEAD)
         / 
A -- B (main)

```

##### Example: `git commit fixup` + `git rebase`
1. `A -- B -- C -- D -- E (HEAD)`
2. `A -- B -- C -- D -- E -- F (HEAD)`
3. `A -- B’ -- C -- D -- E (HEAD)`

#### `reset` vs `reverse`

* `git reset HASH` or `git reset HEAD~N`
Undo local changes by moving pointer back to the specific commit. This means that it changes the 
commit history and is not recommend to be used for public commits.

`--mixed` (default option) makes it so that only the pointer is moved back. This can be used to 
squash your latest commits.

`--hard` flag makes it so that the working directory is also updated.

* `git revert HASH` or `git revert HEAD~N`
Undo changes by adding a new commit that undoes all changes. This means that it does not change the 
commit history and is the recommended way of correcting public commits.

> Both of these commands can also be executed on a file level.

#### `git rev-list HASH..HEAD`

Lists commit objects in reverse chronological order since commit hash. 


## Useful resources

Want some tips on how to fix your Git mistakes and keep your commit history clean? 
Check [ohshitgit](https://ohshitgit.com).
