# ============================================================================
# Purpose:      Node and edge tables of the Numbat clone trees (mutation graph of iteration 2), one pair per tumor
#               (Methods, Copy number variant detection). For every tumor folder of data/amp1q/cnv/numbat_outs that
#               holds mut_graph_2.rds, the node table lists clone, GT, label, root, leaf, depth (breadth-first distance
#               from node 1) and parent (as a clone number); the edge table lists from and to (as clone numbers),
#               from_label and to_label.
# Inputs:       data/amp1q/cnv/numbat_outs/<tumor_name>/mut_graph_2.rds (Numbat, 03_run_numbat.R).
# Outputs:      data/cnv/mutgraph/<tumor_name>.node.tsv and data/cnv/mutgraph/<tumor_name>.edge.tsv.
# Dependencies: R with tidygraph; sources ../config.R.
# ============================================================================
source("../config.R")      # data_path()
suppressPackageStartupMessages(library(tidygraph))

OUT_DIR <- data_path("cnv", "mutgraph")
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

write_graph_files <- function(mut_graph_file, out_dir = OUT_DIR) {
  # tumor name = name of the Numbat output folder
  tumor_name <- basename(dirname(mut_graph_file))

  # Numbat mutation graph with root, leaf, depth and parent of every node
  G <- readRDS(mut_graph_file)
  G_df <- G %>% as_tbl_graph() %>% mutate(root = node_is_root(),
                                          leaf = node_is_leaf(),
                                          depth = bfs_dist(root = 1),
                                          parent = bfs_parent())
  # node table; node IDs converted to clone numbers
  node_df <- G_df %>% activate(nodes) %>% data.frame()
  clone_id <- as.integer(node_df$clone)
  names(clone_id) <- as.integer(node_df$id)

  node_df$parent <- clone_id[node_df$parent]
  node_df <- node_df[c("clone", "GT", "label", "root", "leaf", "depth", "parent")]
  write.table(node_df, file.path(out_dir, paste0(tumor_name, ".node.tsv")),
              quote = FALSE, sep = "\t", row.names = FALSE, col.names = TRUE)

  # edge table with clone numbers
  edge_df <- G_df %>% activate(edges) %>% data.frame()
  edge_df$to <- clone_id[edge_df$to]
  edge_df$from <- clone_id[edge_df$from]
  edge_df <- edge_df[c("from", "to", "from_label", "to_label")]
  write.table(edge_df, file.path(out_dir, paste0(tumor_name, ".edge.tsv")),
              quote = FALSE, sep = "\t", row.names = FALSE, col.names = TRUE)
}

tumor_dirs <- list.files(path = data_path("amp1q", "cnv", "numbat_outs"), full.names = TRUE)

processed <- 0
for (tumor_dir in tumor_dirs) {
  mut_graph_file <- file.path(tumor_dir, "mut_graph_2.rds")
  if (file.exists(mut_graph_file)) {
    write_graph_files(mut_graph_file)
    processed <- processed + 1
  }
}
cat("Tumors with a mutation graph:", processed, "\n")
