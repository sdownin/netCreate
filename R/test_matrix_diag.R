gen_dist <- gen_dist[upper.tri(gen_dist,diag = T)]

age_dist <- sim_array[,,"age_dist"]
age_dist_tri <- age_dist
age_dist_tri[upper.tri(age_dist_tri)] <- NA
age_vec <- c(age_dist_tri[!is.na(age_dist_tri) | age_dist_tri ==0])


point_sim <- sim_array[,,"point_sim"]
point_sim_tri <- point_sim
point_sim_tri[upper.tri(point_sim_tri)] <- NA
point_vec <- c(point_sim_tri[!is.na(point_sim_tri) & point_sim_tri > 0])


point_sim <- sim_array[,,"point_sim"]
point_sim_tri <- point_sim
point_sim_tri[upper.tri(point_sim_tri)] <- NA
point_vec <- c(point_sim_tri[!is.na(point_sim_tri) & point_sim_tri > 0])

ord_per_yr_dist <- sim_array[,,"ord_per_yr_dist"]
ord_per_yr_dist_tri <- ord_per_yr_dist
ord_per_yr_dist_tri[upper.tri(ord_per_yr_dist_tri)] <- NA
ord_per_yr_vec <- c(ord_per_yr_dist_tri[!is.na(ord_per_yr_dist_tri) & ord_per_yr_dist_tri >= 0])
