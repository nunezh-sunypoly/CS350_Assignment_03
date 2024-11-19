# Assignment 3
• Assignment 3 is an analysis of the fraud detection database, for the possibility of second-party fraud
• you are to make a Jupyter notebook for the analysis, in Python
• the procedure is exactly as in previous assignments and
project 1:
  GitHub repo, invitation to collaborate, etc.
Step 1
  make sure the fraud detection database is made in your Neo4j Desktop, with the APOC and GDS plugins installed; then, start (activate) the database, of course
Step 2
  in a directory, where you will have the Jupyter notebook of the assignment, activate your conda environment then, start the Jupyter server with the usual jupyter notebook
Step 3
  at the beginning, one of your team members will create an empty notebook, of course but, usually you will just pop the existing notebook
Step 4
  write the appropriate notebook cells that will allow you to connect to the Neo4j server and to use Neo4j GDS note: refer to the demo in CLASS 18, if necessary
Step 5
  create the SHARED_PII relationship (between clients) and the resulting subgraph note: I already showed the Cypher script for doing it in CLASS 19
  • of course, at every step, you may check, e.g. using Neo4j Browser, that what you specified has been carried out
  • for example, after STEP 5, using CALL db.schema.visualization(), you should see that there a relationship between Client and itself whose type name is SHARED_PII
Step 6
  make the in-memory projection of the graph in STEP 5; use the name 'clientClusters' for this projection; the nodes to be projected are the nodes with label Client; 
  the relationships to be projected are the ones with type SHARED_PII note: I already showed the Cypher script for doing it in CLASS 19
  • but, make sure you understand how to project (a so-called native projection) as a check of STEP 6, once it has been done, you may use the following command in Neo4J Browser
    CALL gds.graph.list()
  which shows all (and only) the in-memory projection graphs that exist for the database
  • it should show only the projection
    ‘clientClusters'
  of course one more remark about YIELD
  • I already explain, in CLASS 19, what exactly YIELD does
  • now, all GDS procedures YIELDs their results; so, for example, you must first use YIELD before you can use RETURN; refer to my demo in CLASS 19 again
  • of course, before you RETURN, you may want to do more things with those YIELDed results
Step 7
  use the WCC (Weakly Connected Components) algorithm, in stream mode, to identify clusters of Client nodes in the above projection graph;
  Hint: look at the WCC documentation again, at
  https://neo4j.com/docs/graph-data-science/current/algorithms/wcc/
  • here, the configuration should be
    {
      nodeLabels: ['Client'],
      relationshipTypes: ['SHARED_PII'],
      consecutiveIds: true
    }
  now, according to the documentation, your code should YIELD nodeId, componentId but there is a catch:
      (a) nodeId is the internal id used by Neo4j to identify the node, not the user-defined id in the database
      (b) componentId is just a consecutive integer to identify each cluster
  • so, you would want something like the following at the end:
      RETURN gds.util.asNode(nodeId).id AS clientId, componentId AS clusterId
  asNode(nodeId) converts nodeId to a Node in the database, and you then access the user-defined id
  • DISPLAYING RESULTS IN THE NOTEBOOK
  your code will not display anything in the Jupyter notebook, unless you do something similar to what I show in the CLASS 18 demo
  • that is, use driver.session() to run the code, and convert the results to a Pandas data frame see that CLASS 18 demo
Step 8
  the purpose of this step is to mark each client that belongs to a cluster of size at least 2 as possibly (not provably) belonging to a fraud ring;
  an isolated client (in a cluster by itself) is again possibly (not provably) not a fraudster
  • so use the same WCC algorithm as in STEP 7, but the end of your code should be like what I show on the next slide
      YIELD nodeId, componentId
      WITH gds.util.asNode(nodeId) AS clientId , componentId AS clusterId
      WITH clusterId, collect(clientId.id) AS clients
      WITH clusterId, clients, size(clients) AS clusterSize WHERE clusterSize >= 2
      UNWIND clients AS client
      MATCH (c:Client) WHERE c.id = client
      SET c.secondPartyFraudRing = clusterId
• See it?
• as I already mentioned in previous classes, WITH is really justas in SQL so, for these non-isolated clients, there will be a new property
      secondPartyFraudRing
whose value is the id of the cluster they belong to
• now, the next step (STEP 9) is about making a graph usable for comparing two clients
• to repeat something I already mentioned before, but which is fundamental in graph data science, imagine persons A, B, C buying items X, Y, Z, T

at least intuitively, we would say that A is more similar to B than to C (they have similar needs/tastes)
• right?
we have a similar situation with our fraud database
• A, B, C are clients; Y=Phone, Z = Email, T = SSN
• see it?
Step 9
this step is really then about making the bipartite graph, as above, so that we can compare similarity between clients, based on how many PIIs they share;
• since this involves a so-called Cypher projection, not the native projection I have used above and my previous demos, I will actually provide code that will do this STEP 9 next slides
      // first, find clients
      MATCH (c:Client) WHERE c.secondPartyFraudRing is NOT NULL
      WITH collect(c) as clients
      // second, find the PII nodes
      MATCH (n) WHERE n:Email OR n:Phone OR n:SSN
      // combine the two sets of nodes
      WITH clients, collect(n) AS piis
      WITH clients + piis AS nodes
      // use only the clients that belong to a cluster of size >= 2
      // as per STEP 8
      MATCH (c:Client) -[:HAS_EMAIL|HAS_PHONE|HAS_SSN]->(p)
      WHERE c.secondPartyFraudRing is NOT NULL
     // now make the bipartite graph,
     // with a new relationship named HAS_PII
     WITH nodes, collect({source: c, target: p}) as relationships
     // use a Cypher projection
     // not the usual native projection
     CALL gds.graph.project.cypher(
      'similarity',
     "UNWIND $nodes as n
     RETURN id(n) AS id,labels(n) AS labels",
     "UNWIND $relationships as r
     RETURN id(r['source']) AS source, id(r['target']) AS target,
     'HAS_PII' as type",
      { parameters:
        { nodes: nodes,
        relationships: relationships }
      })
      YIELD graphName, nodeCount, relationshipCount
      RETURN graphName, nodeCount, relationshipCount

  • refer to my in-class remarks
  • the reason why it is called Cypher projection is that the nodes and relationships to be projected are not given by an explicit lists (as in native projections), but by Cypher statements, here
  namely those UNWIND ...
  • note that $ in $nodes signifies parameters to be given later, in the { parameters: ....} block
  • here, the in-memory projection is given the name 'similarity'
Step 10
  now that we have the above 'similarity' bipartite graph, we can use it to compute similarity score between any pairs of clients of interest; use the nodeSimilarity algorithm, in mutate mode
  Note: refer to the nodeSimilarity doc again, at
  https://neo4j.com/docs/graph-data-science/current/algorithms/node-similarity/
  • here the configuration would be something like
      { mutateProperty: 'jaccardScore',
          mutateRelationshipType: 'SIMILAR_TO' ,
          topK: 15
      }
  recall that Neo4j GDS algorithms may be used in one of three modes: stream, mutate, write stream mode results are shown on the screen (e.g. in Neo4j Browser) and the in-memory projection and the original
  database are not affected (not changed)
mutate mode
  the in-memory database is modified by the results but not the original database
write mode
  the original database is modified using the results about the configuration I provide above, note that gds.nodeSimilarity.mutate() creates a new relationship called 'SIMILAR_TO', and the computed score 
  will be named 'jaccardScore' the clause topK: 15 simply means that each node will have relationships created only with its 15 most similar nodes based on the similarity creating too many relationships 
  for each node may make the graph too convoluted and not useful for analysis
Step 11
  we need, however, to write the 'SIMILAR_TO' relationship back to the original database, with the 'jaccardScore' as a property of the relationship use gds.graph.writeRelationship for this, with the appropriate arguments
  Note: take a look at the doc at
  https://neo4j.com/docs/graph-data-science/current/management-ops/graph-write-to-neo4j/write-back-relationships/
Step 12
  in this step, we want to see the most ‘popular’ nodes, by computing, for each node, the node degree = how many nodes it is connected to;
  here how many nodes a given node is SIMILAR_TO use gds.degree.write whose doc is at
  https://neo4j.com/docs/graph-data-science/current/algorithms/degree-centrality
  
  the configuration should be
      { nodeLabels: ['Client'],
          relationshipTypes: ['SIMILAR_TO'],
          relationshipWeightProperty: 'jaccardScore',
          writeProperty: 'secondPartyFraudScore'
      }
  it creates a new property named secondPartyFraudScore for each client, whose value consists of the totality of the scores on each relationship involving the client
Step 13
  but, we want to label a client as a potential fraudster only if its degree of centrality, as computed in STEP 12, is high enough for these clients, 
  we want to create a new (Boolean) property named SecondPartyFraudster
  • see the next slide for how this may be done
  
  MATCH (c:Client)
  WHERE c.secondPartyFraudScore IS NOT NULL
  WITH percentileCont(c.secondPartyFraudScore, 0.95) AS threshold
  MATCH (c:Client)
  WHERE c.secondPartyFraudScore > threshold
  SET c:SecondPartyFraudster

  • the code should be clear enough
Step 14
finally, we want to list the names and ids of these potential fraudsters you must use Pandas dataframes in the notebook, 
because I want to see the results right there on the notebook see CLASS 18 demo again, if necessary
• here you just MATCH for clients where that property SecondPartyFraudster is set, that is
  MATCH (c:Client)
  WHERE c:SecondPartyFraudster
  and return c’s name and c’s id
  the above is a lot to internalize
• however note that these days we have realized that studying relationships, whether between humans, or between humans and non-humans, etc. is
fundamental in solving not just technical problems but also societal problems
• after all, much of information and knowledge is in relationships (technically, in graphs)
























