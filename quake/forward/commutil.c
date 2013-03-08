/* -*- C -*- */

/* @copyright_notice_start
 *
 * This file is part of the CMU Hercules ground motion simulator developed
 * by the CMU Quake project.
 *
 * Copyright (C) Carnegie Mellon University. All rights reserved.
 *
 * This program is covered by the terms described in the 'LICENSE.txt' file
 * included with this software package.
 *
 * This program comes WITHOUT ANY WARRANTY; without even the implied warranty
 * of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * 'LICENSE.txt' file for more details.
 *
 *  @copyright_notice_end
 */

/*
 * Description: Miscellaneous support functions.
 */
#include <string.h>
#include "commutil.h"
#include "util.h"


/**
 * Broadcast a string to all processes in a MPI communicator from the PE
 * with the given rank.
 *
 * \param string    Character string to broadcast.  The root PE passes an
 *		    initialized string.  The receiving PEs will allocate
 *		    memory for the received string.
 * \param root_rank Rank (i.e, PE ID) of the process sending the string.
 *                  That is, root of the broadcast.
 * \param comm      MPI communicator structure.
 *
 * \return 0 on success, -1 on error (really abort through \c solver_abort() ).
 *
 * \author jclopez
 */
int
broadcast_string( char** string, int root_rank, MPI_Comm comm )
{
    int32_t string_len = 0;
    int32_t myID;

    HU_ASSERT_PTR( string );

    MPI_Comm_rank( comm, &myID );

    if (myID == root_rank) {
	HU_ASSERT_PTR( *string );
	string_len = strlen( *string );
    }

    MPI_Bcast( &string_len, 1, MPI_INT, root_rank, comm );

    if (myID != root_rank) {
	XMALLOC_STRING( (*string), string_len );
    }

    /* include terminating \0 character in the broadcast */
    MPI_Bcast( *string, string_len + 1, MPI_CHAR, root_rank, comm );

    (*string)[string_len] = '\0';	/* paranoid safeguard */

    return 0;
}

/**
 * Broadcast a char array (e.g., a string) of a given length.  The memory
 * must be pre-allocated by the caller both at the sender and receiver.
 * The destination buffer (on the receiver side must be of size equal or
 * larger than the length specified on the sender side.
 */
int
broadcast_char_array( char string[], size_t len, int root_rank, MPI_Comm comm )
{
    int32_t sender_len = len;

    /* pass the length of the array to send */
    MPI_Bcast( &sender_len, 1, MPI_INT, root_rank, comm );
    HU_ASSERT_ALWAYS( sender_len <= len );

    /* broadcast the actual char array */
    MPI_Bcast( string, sender_len, MPI_CHAR, root_rank, comm );

    return 0;
}
