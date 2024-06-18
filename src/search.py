# def exhaustive_search(text: List[str], num_authors, model, tokenizer, device):

#     authors = [list(range(num_authors)) for _ in range(len(text))]
#     authors = list(product(*authors))
#     authors = [author for author in authors if len(np.unique(author)) == num_authors]
    
#     changes_to_score =  {}
#     for author in authors:
#         proposal_score = get_proposal_score(text, author, model, tokenizer, device)

#         # key = (author, tuple(get_changes(author)))
#         key = tuple(get_changes(author))
#         changes_to_score[key] = proposal_score

#     return changes_to_score
    
