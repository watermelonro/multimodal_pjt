import wave
import io
import struct
import logging

# --- 로깅 설정 ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def _split_buffer_into_wav_chunks(buffer_bytes):
    """Concatenated WAV bytes into a list of individual WAV bytes."""
    chunks = []
    offset = 0
    while offset < len(buffer_bytes):
        # Check for RIFF header
        if offset + 8 > len(buffer_bytes) or buffer_bytes[offset:offset+4] != b'RIFF':
            logger.warning(f"No RIFF header found at offset {offset}. Stopping parse.")
            break
        
        try:
            # Read the size of the chunk from the header
            chunk_size = struct.unpack('<I', buffer_bytes[offset+4:offset+8])[0]
            total_chunk_size = chunk_size + 8
            
            if offset + total_chunk_size > len(buffer_bytes):
                logger.warning("Incomplete chunk found at the end of the buffer.")
                # Add the rest of the buffer as the last chunk, even if incomplete
                chunks.append(buffer_bytes[offset:])
                break

            chunk = buffer_bytes[offset : offset + total_chunk_size]
            chunks.append(chunk)
            offset += total_chunk_size
        except struct.error as e:
            logger.error(f"Error parsing chunk size at offset {offset}: {e}")
            break
            
    logger.info(f"Split buffer into {len(chunks)} WAV chunks.")
    return chunks

def merge_wav_chunks_from_buffer(buffer_bytes):
    """
    Merges multiple WAV chunks from a single byte buffer into a single WAV byte string.
    This version is robust and uses the 'wave' module for correctness.
    """
    if not buffer_bytes:
        logger.warning("Input buffer is empty.")
        return b''

    wav_chunks = _split_buffer_into_wav_chunks(buffer_bytes)
    
    if not wav_chunks:
        logger.error("Could not find any valid WAV chunks in the buffer.")
        return b'' # Return empty bytes if no chunks found

    output_buffer = io.BytesIO()
    
    try:
        # Use the first chunk to get the audio parameters
        with wave.open(io.BytesIO(wav_chunks[0]), 'rb') as first_wav:
            params = first_wav.getparams()
            # nchannels, sampwidth, framerate, nframes, comptype, compname
            logger.info(f"Using WAV parameters from first chunk: {params}")

        # Open the output wave file for writing
        with wave.open(output_buffer, 'wb') as output_wav:
            output_wav.setparams(params)
            
            # Loop through all chunks, read their frames, and write to the output
            for i, chunk in enumerate(wav_chunks):
                try:
                    with wave.open(io.BytesIO(chunk), 'rb') as wav_file:
                        output_wav.writeframes(wav_file.readframes(wav_file.getnframes()))
                except wave.Error as e:
                    logger.warning(f"Could not process WAV chunk #{i}. It might be corrupted. Skipping. Error: {e}")
                    continue

        final_wav_bytes = output_buffer.getvalue()
        logger.info(f"Successfully merged {len(wav_chunks)} chunks into a single WAV file of {len(final_wav_bytes)} bytes.")
        return final_wav_bytes

    except wave.Error as e:
        logger.error(f"Fatal error during WAV merging process: {e}")
        return b'' # Return empty on fatal error
    except Exception as e:
        logger.error(f"An unexpected error occurred during merging: {e}")
        return b''
