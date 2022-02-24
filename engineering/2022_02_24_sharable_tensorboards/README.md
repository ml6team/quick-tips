# Vertex AI TensorBoard alternative for smaller budgets

At the time of writing the pricing plan for Vertex AI TensorBoard is the following: 
> Vertex AI TensorBoard charges a monthly fee of $300 per unique active user. Active users are measured through the Vertex AI TensorBoard UI.

This can get pretty expensive if you want to share your experiment tracking results with your team...

## Any other alternatives? 🤷‍♂️

Some other options (like TensorBoard.dev)  make your data publicly available (it goes without saying but this is a big nono when working with customer data 🙅‍♂️)
Other solutions require other people to set up a local environment to run TensorBoard and thus require certain technical skills from those people.

## Our solution 🦾

You can use the Terraform code in the `tensorboard.tf` file to generate your own securely sharable TensorBoard for a fraction of the price. 

Some minor changes have to be made to the file: 

- change the support email for the `oauth_consent_screen`
- add the allowed users in the `allowed_users` resource

You only need to run the classic `terraform apply` and everything will be set up for you. (Don't forget to pass the project name as a parameter)

That's it! 🎉 

You can now start writing logs to the bucket you specified and inspect it through your own sharable TensorBoard in a cost-effective way! 

Want to learn more about our approach? We have a two-part blog post about it: 

* [Part 1](https://blog.ml6.eu/a-vertex-ai-tensorboard-alternative-for-smaller-budgets-part-1-ab840d2a592a) - Some context about the problem
* [Part 2](https://blog.ml6.eu/a-vertex-ai-tensorboard-alternative-for-smaller-budgets-part-2-923953c1e422) - The solution

Keep on experiment tracking! ⚡️