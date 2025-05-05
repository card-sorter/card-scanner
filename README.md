# card-scanner

flowchart 
    A[New Card Image] -> B[Generate pHash (64 bits)]
    B -> C[Search KDTree of Stored pHashes]
    C -> D[Find Nearest Card Match]
    D -> E[Check Similarity Score]
    E -> |High Match| F[Assign to Correct Bin]
    E -> |Low Match| G[Send for Manual Review]